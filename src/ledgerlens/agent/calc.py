"""Safe arithmetic for the `calculate` tool.

The expression is parsed with `ast` and evaluated by walking an allow-list of node
types. Nothing is ever passed to eval/exec, names other than a few math functions are
rejected, and exponent and input sizes are bounded so a model cannot hang the process.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable

MAX_LEN = 300
MAX_EXPONENT = 12
MAX_MAGNITUDE = 1e18

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": lambda x, digits=0: round(x, int(digits)),
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


class CalcError(ValueError):
    """The expression is not plain arithmetic or cannot be evaluated."""


def _clean(expression: str) -> str:
    """Accept what models naturally write: thousands separators, $, x for times, and %
    as percent (so modulo is not supported)."""
    expr = expression.strip()
    # Thousands separators only inside a grouped integer part, so "round(2.5,100)" keeps
    # its argument comma.
    expr = re.sub(
        r"(?<![\d.])\d{1,3}(?:,\d{3})+(?![\d,])", lambda m: m.group(0).replace(",", ""), expr
    )
    expr = expr.replace("$", "").replace("×", "*").replace("−", "-")
    expr = re.sub(r"(?<=\d)\s*%", "/100", expr)
    expr = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", expr)
    return expr


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        if isinstance(node.value, bool):
            raise CalcError("booleans are not numbers")
        value = float(node.value)
        if not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
            raise CalcError("number out of range")
        return value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalcError(f"exponent larger than {MAX_EXPONENT}")
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalcError("division by zero")
        try:
            result = _BINARY[type(node.op)](left, right)
        except (OverflowError, ZeroDivisionError) as exc:
            raise CalcError("result out of range") from exc
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        result = _UNARY[type(node.op)](_eval(node.operand))
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        args = [_eval(a) for a in node.args]
        try:
            result = float(_FUNCS[node.func.id](*args))
        except (TypeError, ValueError) as exc:
            raise CalcError(f"bad arguments to {node.func.id}") from exc
    else:
        raise CalcError(f"unsupported syntax: {type(node).__name__}")
    if isinstance(result, complex) or not math.isfinite(result) or abs(result) > MAX_MAGNITUDE:
        raise CalcError("result out of range")
    return float(result)


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression such as ``(391.0 - 383.3) / 383.3 * 100``."""
    if len(expression) > MAX_LEN:
        raise CalcError(f"expression longer than {MAX_LEN} characters")
    try:
        tree = ast.parse(_clean(expression), mode="eval")
    except SyntaxError as exc:
        raise CalcError("not a valid arithmetic expression") from exc
    return _eval(tree)
