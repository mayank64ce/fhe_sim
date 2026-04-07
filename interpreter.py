"""
Symbolic interpreter for OpenFHE eval() methods.

Walk the tree-sitter AST of a C++ eval() function body:
  - Constant-fold arithmetic expressions
  - Unroll for-loops whose bounds are statically known
  - Track variable types  (Ciphertext / Plaintext / plain)
  - Track ciphertext levels through operations
  - Collect a FHEOp log: [(op_type, input_level), ...]
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Node

from .types import FHEType, FHEOp, OpType, FHE_METHOD_RETURN_TYPE


# ---------------------------------------------------------------------------
# Tree-sitter setup
# ---------------------------------------------------------------------------

_CPP_LANGUAGE = Language(tscpp.language())

def _make_parser() -> Parser:
    return Parser(_CPP_LANGUAGE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(node: Node) -> str:
    return node.text.decode("utf-8")


_BUILTINS: dict[str, Any] = {
    "pow":   lambda a, b: a ** b,
    "log2":  lambda a: math.log2(a),
    "log":   lambda a: math.log(a),
    "sqrt":  lambda a: math.sqrt(a),
    "abs":   lambda a: abs(a),
    "ceil":  lambda a: math.ceil(a),
    "floor": lambda a: math.floor(a),
}


# ---------------------------------------------------------------------------
# Execution context (replaces three separate dicts threaded everywhere)
# ---------------------------------------------------------------------------

@dataclass
class ExecContext:
    """
    Carries all mutable interpreter state through the AST walk.

    env       : name → scalar value  (int/float, for constant folding)
    type_env  : name → FHEType
    level_env : name → int  (ciphertext level; absent = initial_level)
    """
    env:       dict = field(default_factory=dict)
    type_env:  dict = field(default_factory=dict)
    level_env: dict = field(default_factory=dict)

    def copy(self) -> "ExecContext":
        return ExecContext(
            env       = dict(self.env),
            type_env  = dict(self.type_env),
            level_env = dict(self.level_env),
        )

    def merge_from(self, child: "ExecContext"):
        """Pull all changes from a child context back into self."""
        self.env.update(child.env)
        self.type_env.update(child.type_env)
        self.level_env.update(child.level_env)


# ---------------------------------------------------------------------------
# FHE type from C++ type string
# ---------------------------------------------------------------------------

def _fhe_type_from_cpp(type_str: str) -> FHEType:
    if "Ciphertext" in type_str:
        return FHEType.CIPHERTEXT
    if "Plaintext" in type_str:
        return FHEType.PLAINTEXT
    return FHEType.PLAIN


# ---------------------------------------------------------------------------
# Member-type extraction from .h file
# ---------------------------------------------------------------------------

def extract_member_types(header_src: str) -> dict[str, FHEType]:
    parser = _make_parser()
    tree = parser.parse(header_src.encode())
    result: dict[str, FHEType] = {}

    def walk(node: Node):
        if node.type == "field_declaration":
            type_node = node.child_by_field_name("type")
            type_str = _text(type_node) if type_node else ""
            fhe_t = _fhe_type_from_cpp(type_str)
            if fhe_t != FHEType.PLAIN:
                for child in node.children:
                    if child.type == "field_identifier":
                        result[_text(child)] = fhe_t
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return result


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class Interpreter:
    """
    Symbolically executes the eval() method of an OpenFHE solution file.
    Produces a list of FHEOp(op_type, level) in execution order.
    """

    def __init__(
        self,
        member_types:            dict[str, FHEType],
        initial_level:           int = 10,
        levels_after_bootstrap:  int = 10,
    ):
        self._initial_level          = initial_level
        self._levels_after_bootstrap = levels_after_bootstrap

        # Seed global type/level envs from class header
        self._global_type_env: dict[str, FHEType] = dict(member_types)
        self._global_level_env: dict[str, int] = {
            name: initial_level
            for name, t in member_types.items()
            if t == FHEType.CIPHERTEXT
        }

        self.op_log: list[FHEOp] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, eval_src: str) -> list[FHEOp]:
        parser = _make_parser()
        tree = parser.parse(eval_src.encode())

        body = self._find_eval_body(tree.root_node)
        if body is None:
            raise ValueError("Could not find eval() function definition in source.")

        ctx = ExecContext(
            type_env  = dict(self._global_type_env),
            level_env = dict(self._global_level_env),
        )
        self._exec_compound(body, ctx)
        return self.op_log

    # ------------------------------------------------------------------
    # AST helpers
    # ------------------------------------------------------------------

    def _find_eval_body(self, root: Node) -> Optional[Node]:
        for node in self._walk(root):
            if node.type == "function_definition":
                decl = node.child_by_field_name("declarator")
                if decl and self._function_name(decl) == "eval":
                    return node.child_by_field_name("body")
        return None

    @staticmethod
    def _function_name(func_decl: Node) -> Optional[str]:
        if func_decl.type != "function_declarator":
            return None
        name_node = func_decl.children[0] if func_decl.children else None
        if name_node is None:
            return None
        if name_node.type == "qualified_identifier":
            ident = name_node.child_by_field_name("name")
            return _text(ident) if ident else None
        if name_node.type == "identifier":
            return _text(name_node)
        return None

    @staticmethod
    def _walk(node: Node):
        yield node
        for c in node.children:
            yield from Interpreter._walk(c)

    # ------------------------------------------------------------------
    # Statement executor
    # ------------------------------------------------------------------

    def _exec_compound(self, node: Node, ctx: ExecContext):
        for child in node.children:
            self._exec_stmt(child, ctx)

    def _exec_stmt(self, node: Node, ctx: ExecContext):
        t = node.type
        if t == "compound_statement":
            self._exec_compound(node, ctx)
        elif t == "declaration":
            self._exec_declaration(node, ctx)
        elif t == "expression_statement":
            expr = next((c for c in node.children if c.is_named), None)
            if expr is not None:
                self._eval_expr(expr, ctx)
        elif t == "for_statement":
            self._exec_for(node, ctx)
        elif t == "if_statement":
            self._exec_if(node, ctx)

    # ------------------------------------------------------------------
    # Declaration handler
    # ------------------------------------------------------------------

    def _exec_declaration(self, node: Node, ctx: ExecContext):
        type_node = node.child_by_field_name("type")
        type_str  = _text(type_node) if type_node else ""
        fhe_t     = _fhe_type_from_cpp(type_str)

        for child in node.children:
            if child.type == "init_declarator":
                decl      = child.child_by_field_name("declarator")
                val_node  = child.child_by_field_name("value")
                name      = self._extract_var_name(decl)
                if name:
                    ctx.type_env[name] = fhe_t
                    if fhe_t == FHEType.CIPHERTEXT:
                        ctx.level_env[name] = self._initial_level
                    if val_node is not None:
                        val, vt, vlvl = self._eval_expr(val_node, ctx)
                        if val is not None and fhe_t == FHEType.PLAIN:
                            ctx.env[name] = val
                        if vt not in (FHEType.PLAIN, FHEType.UNKNOWN):
                            ctx.type_env[name] = vt
                        if vt == FHEType.CIPHERTEXT and vlvl is not None:
                            ctx.level_env[name] = vlvl

            elif child.type == "identifier":
                name = _text(child)
                ctx.type_env[name] = fhe_t
                if fhe_t == FHEType.CIPHERTEXT:
                    ctx.level_env[name] = self._initial_level

    def _extract_var_name(self, node: Optional[Node]) -> Optional[str]:
        if node is None:
            return None
        if node.type == "identifier":
            return _text(node)
        inner = node.child_by_field_name("declarator")
        if inner:
            return self._extract_var_name(inner)
        return None

    # ------------------------------------------------------------------
    # For-loop unroller
    # ------------------------------------------------------------------

    def _exec_for(self, node: Node, ctx: ExecContext):
        init_node   = node.child_by_field_name("initializer")
        cond_node   = node.child_by_field_name("condition")
        update_node = node.child_by_field_name("update")
        body_node   = node.child_by_field_name("body")

        var, start = self._parse_for_init(init_node, ctx)
        end, incl  = self._parse_for_cond(cond_node, var, ctx)
        step       = self._parse_for_update(update_node, var)

        if var is None or start is None or end is None or step is None:
            self._exec_stmt(body_node, ctx)
            return

        end_val   = int(end) + (1 if incl else 0)
        start_val = int(start)
        step_val  = int(step)

        if step_val == 0:
            return
        if step_val > 0 and start_val >= end_val:
            return
        if step_val < 0 and start_val <= end_val:
            return

        for i in range(start_val, end_val, step_val):
            child_ctx = ctx.copy()
            child_ctx.env[var] = i
            self._exec_stmt(body_node, child_ctx)
            # Propagate all changes back (levels, types, values)
            ctx.merge_from(child_ctx)

    def _parse_for_init(self, node: Optional[Node], ctx: ExecContext):
        if node is None:
            return None, None
        if node.type == "declaration":
            self._exec_declaration(node, ctx)
            for child in node.children:
                if child.type == "init_declarator":
                    decl     = child.child_by_field_name("declarator")
                    val_node = child.child_by_field_name("value")
                    name     = self._extract_var_name(decl)
                    if val_node is not None:
                        val, _, _ = self._eval_expr(val_node, ctx)
                        return name, val
        return None, None

    def _parse_for_cond(self, node: Optional[Node], var: Optional[str],
                        ctx: ExecContext):
        if node is None or var is None:
            return None, False
        if node.type == "binary_expression":
            op    = _text(node.child_by_field_name("operator"))
            left  = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left and _text(left) == var and right:
                val, _, _ = self._eval_expr(right, ctx)
                if val is not None:
                    return val, op == "<="
        return None, False

    def _parse_for_update(self, node: Optional[Node], var: Optional[str]):
        if node is None or var is None:
            return None
        txt = _text(node).strip()
        if txt in (f"{var}++", f"++{var}"):
            return 1
        if txt in (f"{var}--", f"--{var}"):
            return -1
        m = re.match(rf"{re.escape(var)}\s*\+=\s*(.+)", txt)
        if m:
            try:
                return int(eval(m.group(1), {"__builtins__": {}}, {}))
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # If-statement handler
    # ------------------------------------------------------------------

    def _exec_if(self, node: Node, ctx: ExecContext):
        cond_node   = node.child_by_field_name("condition")
        consequence = node.child_by_field_name("consequence")
        alternative = node.child_by_field_name("alternative")

        cond_val = self._eval_condition(cond_node, ctx)

        if cond_val is True:
            self._exec_stmt(consequence, ctx)
        elif cond_val is False:
            if alternative is not None:
                else_body = next((c for c in alternative.children if c.is_named), None)
                if else_body:
                    self._exec_stmt(else_body, ctx)
        else:
            # Unknown condition — execute both branches
            self._exec_stmt(consequence, ctx)
            if alternative is not None:
                else_body = next((c for c in alternative.children if c.is_named), None)
                if else_body:
                    self._exec_stmt(else_body, ctx)

    def _eval_condition(self, node: Optional[Node], ctx: ExecContext) -> Optional[bool]:
        if node is None:
            return None
        if node.type == "condition_clause":
            inner = node.child_by_field_name("value")
            return self._eval_condition(inner, ctx)
        if node.type == "parenthesized_expression":
            inner = next((c for c in node.children if c.is_named), None)
            return self._eval_condition(inner, ctx)
        if node.type == "binary_expression":
            op = _text(node.child_by_field_name("operator"))
            lv, _, _ = self._eval_expr(node.child_by_field_name("left"),  ctx)
            rv, _, _ = self._eval_expr(node.child_by_field_name("right"), ctx)
            if lv is None or rv is None:
                return None
            return {"==": lv==rv, "!=": lv!=rv, "<": lv<rv,
                    "<=": lv<=rv, ">": lv>rv, ">=": lv>=rv}.get(op)
        return None

    # ------------------------------------------------------------------
    # Expression evaluator
    # Returns (value_or_None, FHEType, level_or_None)
    # Side-effect: records FHE ops, updates ctx type/level for assignments
    # ------------------------------------------------------------------

    def _eval_expr(self, node: Optional[Node], ctx: ExecContext):
        if node is None:
            return None, FHEType.UNKNOWN, None

        t = node.type

        # ── literals ──────────────────────────────────────────────────
        if t == "number_literal":
            txt = _text(node)
            try:
                return int(txt), FHEType.PLAIN, None
            except ValueError:
                try:
                    return float(txt), FHEType.PLAIN, None
                except ValueError:
                    return None, FHEType.PLAIN, None

        # ── identifiers ───────────────────────────────────────────────
        if t == "identifier":
            name  = _text(node)
            val   = ctx.env.get(name)
            fhe_t = ctx.type_env.get(name,
                    self._global_type_env.get(name, FHEType.PLAIN))
            lvl   = ctx.level_env.get(name,
                    self._global_level_env.get(name)) \
                    if fhe_t == FHEType.CIPHERTEXT else None
            return val, fhe_t, lvl

        # ── parenthesized ─────────────────────────────────────────────
        if t == "parenthesized_expression":
            inner = next((c for c in node.children if c.is_named), None)
            return self._eval_expr(inner, ctx)

        # ── cast: (int)(expr) ─────────────────────────────────────────
        if t == "cast_expression":
            v, ft, lvl = self._eval_expr(node.child_by_field_name("value"), ctx)
            if v is not None:
                try:
                    return int(v), ft, lvl
                except (TypeError, ValueError):
                    pass
            return v, ft, lvl

        # ── unary: -x ─────────────────────────────────────────────────
        if t == "unary_expression":
            op = _text(node.child_by_field_name("operator"))
            v, ft, lvl = self._eval_expr(node.child_by_field_name("argument"), ctx)
            if op == "-" and v is not None:
                return -v, ft, lvl
            return None, ft, lvl

        # ── binary arithmetic ─────────────────────────────────────────
        if t == "binary_expression":
            lv, lt, ll = self._eval_expr(node.child_by_field_name("left"),  ctx)
            rv, rt, rl = self._eval_expr(node.child_by_field_name("right"), ctx)
            fhe_t = lt if lt != FHEType.PLAIN else rt
            lvl   = ll if lt != FHEType.PLAIN else rl
            if lv is not None and rv is not None:
                op = _text(node.child_by_field_name("operator"))
                try:
                    result = {
                        "+": lv + rv, "-": lv - rv,
                        "*": lv * rv,
                        "/": int(lv / rv),
                        "%": int(lv) % int(rv),
                    }.get(op)
                    return result, fhe_t, lvl
                except (ZeroDivisionError, TypeError):
                    pass
            return None, fhe_t, lvl

        # ── assignment: a = expr  /  a[i] = expr ─────────────────────
        if t == "assignment_expression":
            lhs = node.child_by_field_name("left")
            rhs = node.child_by_field_name("right")
            val, fhe_t, lvl = self._eval_expr(rhs, ctx)
            lhs_name = self._lhs_name(lhs)      # base name for type_env
            lvl_key  = self._lhs_level_key(lhs, ctx)  # compound key for level_env
            if lhs_name:
                if fhe_t not in (FHEType.PLAIN, FHEType.UNKNOWN):
                    ctx.type_env[lhs_name] = fhe_t
                if fhe_t == FHEType.PLAIN and val is not None:
                    ctx.env[lhs_name] = val
            if lvl_key and fhe_t == FHEType.CIPHERTEXT and lvl is not None:
                ctx.level_env[lvl_key] = lvl
            return val, fhe_t, lvl

        # ── subscript: ab1[j] ─────────────────────────────────────────
        if t == "subscript_expression":
            arr_node   = node.child_by_field_name("argument")
            indices_n  = node.child_by_field_name("indices")
            idx_node   = (next((c for c in indices_n.children if c.is_named), None)
                          if indices_n else None)
            arr_name = _text(arr_node) if arr_node else None
            fhe_t = (ctx.type_env.get(arr_name,
                     self._global_type_env.get(arr_name, FHEType.PLAIN))
                     if arr_name else FHEType.UNKNOWN)
            lvl = None
            if arr_name and fhe_t == FHEType.CIPHERTEXT:
                # Try per-index key first ("ab1__0"), fall back to array name
                if idx_node:
                    idx_val, _, _ = self._eval_expr(idx_node, ctx)
                    if idx_val is not None:
                        compound = f"{arr_name}__{int(idx_val)}"
                        if compound in ctx.level_env:
                            lvl = ctx.level_env[compound]
                        else:
                            lvl = ctx.level_env.get(arr_name,
                                  self._global_level_env.get(arr_name,
                                  self._initial_level))
                    else:
                        lvl = ctx.level_env.get(arr_name,
                              self._global_level_env.get(arr_name,
                              self._initial_level))
                else:
                    lvl = ctx.level_env.get(arr_name,
                          self._global_level_env.get(arr_name,
                          self._initial_level))
            return None, fhe_t, lvl

        # ── function call ─────────────────────────────────────────────
        if t == "call_expression":
            return self._eval_call(node, ctx)

        # ── comma expression ──────────────────────────────────────────
        if t == "comma_expression":
            val, ft, lvl = None, FHEType.PLAIN, None
            for c in node.children:
                if c.is_named:
                    val, ft, lvl = self._eval_expr(c, ctx)
            return val, ft, lvl

        # ── initializer list ──────────────────────────────────────────
        if t == "initializer_list":
            for c in node.children:
                if c.is_named:
                    self._eval_expr(c, ctx)
            return None, FHEType.PLAIN, None

        return None, FHEType.UNKNOWN, None

    def _lhs_name(self, node: Optional[Node]) -> Optional[str]:
        """Return the base variable name for type_env updates."""
        if node is None:
            return None
        if node.type == "identifier":
            return _text(node)
        if node.type == "subscript_expression":
            arr = node.child_by_field_name("argument")
            return _text(arr) if arr else None
        return None

    def _lhs_level_key(self, node: Optional[Node],
                       ctx: ExecContext) -> Optional[str]:
        """
        Return the most specific key for level_env updates.
        For `ab1[j]` with j=0 in scope → "ab1__0" (per-index tracking).
        Falls back to the base name if index is unknown.
        """
        if node is None:
            return None
        if node.type == "identifier":
            return _text(node)
        if node.type == "subscript_expression":
            arr      = node.child_by_field_name("argument")
            indices_n = node.child_by_field_name("indices")
            idx      = (next((c for c in indices_n.children if c.is_named), None)
                        if indices_n else None)
            arr_name = _text(arr) if arr else None
            if idx and arr_name:
                idx_val, _, _ = self._eval_expr(idx, ctx)
                if idx_val is not None:
                    return f"{arr_name}__{int(idx_val)}"
            return arr_name
        return None

    # ------------------------------------------------------------------
    # Function call evaluation
    # ------------------------------------------------------------------

    def _eval_call(self, node: Node, ctx: ExecContext):
        func_node = node.child_by_field_name("function")
        args_node = node.child_by_field_name("arguments")

        method_name = self._extract_method_name(func_node)

        if method_name and method_name in FHE_METHOD_RETURN_TYPE:
            return self._eval_fhe_call(method_name, args_node, ctx)

        # Math builtins
        func_name = _text(func_node) if func_node else ""
        if func_name in _BUILTINS:
            arg_results = self._eval_arg_list(args_node, ctx)
            vals = [v for v, _, _ in arg_results]
            if all(v is not None for v in vals):
                try:
                    return _BUILTINS[func_name](*vals), FHEType.PLAIN, None
                except Exception:
                    pass
            return None, FHEType.PLAIN, None

        # Unknown call — evaluate args for side-effects
        self._eval_arg_list(args_node, ctx)
        return None, FHEType.UNKNOWN, None

    @staticmethod
    def _extract_method_name(func_node: Optional[Node]) -> Optional[str]:
        if func_node is None:
            return None
        if func_node.type == "field_expression":
            field = func_node.child_by_field_name("field")
            return _text(field) if field else None
        return None

    def _eval_fhe_call(self, method: str, args_node: Optional[Node],
                       ctx: ExecContext):
        arg_results = self._eval_arg_list(args_node, ctx)
        arg_types   = [ft for _, ft, _ in arg_results]
        arg_levels  = [lvl for _, ft, lvl in arg_results
                       if ft == FHEType.CIPHERTEXT and lvl is not None]

        # Input level = minimum level among CT arguments (level matching)
        input_level = (min(arg_levels) if arg_levels
                       else self._initial_level)

        op = self._classify_fhe_op(method, arg_types)
        if op is not None:
            self.op_log.append(FHEOp(op_type=op, level=input_level))

        return_type = FHE_METHOD_RETURN_TYPE.get(method, FHEType.CIPHERTEXT)

        # Compute output level
        if return_type == FHEType.CIPHERTEXT:
            if method in ("EvalMult",):
                output_level = max(0, input_level - 1)
            elif method in ("EvalBootstrap",):
                output_level = self._levels_after_bootstrap
            else:
                output_level = input_level
        else:
            output_level = None

        return None, return_type, output_level

    def _classify_fhe_op(self, method: str,
                         arg_types: list[FHEType]) -> Optional[OpType]:
        ct_args = [t for t in arg_types
                   if t in (FHEType.CIPHERTEXT, FHEType.PLAINTEXT)]

        if method == "EvalMult":
            if all(t == FHEType.CIPHERTEXT for t in ct_args):
                return OpType.EVAL_MULT_CTCT
            return OpType.EVAL_MULT_CTPT

        if method in ("EvalAdd", "EvalSub"):
            if all(t == FHEType.CIPHERTEXT for t in ct_args):
                return OpType.EVAL_ADD_CTCT
            return OpType.EVAL_ADD_CTPT

        if method in ("EvalAddInPlace", "EvalSubInPlace"):
            return OpType.EVAL_ADD_INPLACE

        if method == "EvalRotate":
            return OpType.EVAL_ROTATE

        if method == "EvalBootstrap":
            return OpType.EVAL_BOOTSTRAP

        if method in ("MakeCKKSPackedPlaintext", "MakeBFVPackedPlaintext",
                      "MakePackedPlaintext", "MakePlaintext"):
            return OpType.MAKE_PACKED_PLAINTEXT

        return None

    def _eval_arg_list(self, args_node: Optional[Node],
                       ctx: ExecContext) -> list[tuple]:
        if args_node is None:
            return []
        return [
            self._eval_expr(child, ctx)
            for child in args_node.children
            if child.is_named and child.type != "comment"
        ]
