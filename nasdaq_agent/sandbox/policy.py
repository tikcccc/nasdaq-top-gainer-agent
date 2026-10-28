"""Static checks on agent-generated code before it is executed.

This is defence in depth on top of process isolation: the generated code only
needs pandas/numpy/matplotlib and the three helpers, so everything else
(file and network I/O, introspection, dynamic execution) is rejected.
"""

from __future__ import annotations

import ast

ALLOWED_IMPORTS = {
    "pandas", "numpy", "matplotlib", "matplotlib.pyplot", "matplotlib.dates", "matplotlib.ticker",
    "math", "statistics", "json", "datetime", "collections", "itertools",
}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "os", "sys", "subprocess",
    "socket", "shutil", "pathlib", "importlib", "builtins",
}
# File / network I/O on pandas & numpy objects; charts must go through save_chart().
BANNED_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_feather", "to_hdf", "to_sql", "to_stata",
    "to_html", "to_latex", "to_xml", "to_clipboard", "savefig", "load", "save", "savez", "savetxt",
    "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "system", "popen",
    "io", "DataSource", "get_handle", "fromregex", "ctypes", "ctypeslib", "write", "read", "open",
}
MAX_CODE_CHARS = 20_000


def check(code: str) -> list[str]:
    """Return a list of policy violations (empty when the code may run)."""
    if len(code) > MAX_CODE_CHARS:
        return [f"code is {len(code)} chars; limit is {MAX_CODE_CHARS}"]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    problems.append(f"line {node.lineno}: import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS or node.level:
                problems.append(f"line {node.lineno}: 'from {node.module} import ...' is not allowed")
            for alias in node.names:
                if alias.name.startswith("_") or alias.name == "*" or alias.name in BANNED_NAMES | BANNED_ATTRS:
                    problems.append(f"line {node.lineno}: importing '{alias.name}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                problems.append(f"line {node.lineno}: dunder attribute '{node.attr}' is not allowed")
            elif node.attr in BANNED_ATTRS or node.attr.startswith("read_"):
                problems.append(f"line {node.lineno}: '.{node.attr}' (file/network I/O) is not allowed")
    return sorted(set(problems))
