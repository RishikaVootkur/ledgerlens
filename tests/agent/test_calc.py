import pytest

from ledgerlens.agent.calc import CalcError, calculate


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(391.0 - 383.3) / 383.3 * 100", (391.0 - 383.3) / 383.3 * 100),
        ("-5 + 3", -2),
        ("2 ** 10", 1024),
        ("1,234,567 + 1", 1234568),
        ("$100 x 3", 300),
        ("15%", 0.15),
        ("50% * 200", 100),
        ("abs(-3.5)", 3.5),
        ("round(2.345, 2)", 2.35),
        ("max(1, 9, 4) - min(2, 3)", 7),
        ("sqrt(16)", 4),
        ("7 − 2", 5),
    ],
)
def test_arithmetic(expr, expected):
    assert calculate(expr) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "(lambda: 1)()",
        "[1, 2, 3]",
        "x + 1",
        "abs.__class__",
        "1 if True else 2",
        "9 ** 9 ** 9",
        "10 ** 400",
        "1 / 0",
        "2 +",
        "True + 1",
        "'a' * 3",
        "print(1)",
        "1" * 400,
    ],
)
def test_rejects_non_arithmetic(expr):
    with pytest.raises(CalcError):
        calculate(expr)
