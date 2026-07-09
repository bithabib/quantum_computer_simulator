"""Static (AST-based) validation of user-submitted code.

This is the *first* line of defence. It rejects code before it ever runs by
walking the abstract syntax tree and refusing anything outside a small,
quantum-focused allowlist:

  * imports limited to qsim, numpy, math, cmath, random
  * no access to dunder names (``__class__``, ``__globals__``, ...), which is
    the usual escape hatch out of a restricted namespace
  * no dangerous builtins (eval, exec, open, __import__, getattr, ...)

Static analysis alone is never a complete sandbox, so it is combined with a
restricted-builtins namespace and out-of-process execution with a timeout
(see :mod:`quantum_lab.sandbox`). Defence in depth.
"""

import ast

ALLOWED_IMPORTS = {"qsim", "numpy", "math", "cmath", "random"}

# Names that must never be referenced or called.
FORBIDDEN_NAMES = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "memoryview",
    "breakpoint",
    "help",
    "exit",
    "quit",
    "copyright",
    "credits",
    "license",
}


class ValidationError(Exception):
    """Raised when user code fails static validation."""


class _Validator(ast.NodeVisitor):
    def _fail(self, node, msg):
        line = getattr(node, "lineno", "?")
        raise ValidationError("Line %s: %s" % (line, msg))

    # -- imports ---------------------------------------------------------
    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                self._fail(node, "import of %r is not allowed" % alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        root = (node.module or "").split(".")[0]
        if root not in ALLOWED_IMPORTS:
            self._fail(node, "import from %r is not allowed" % node.module)
        self.generic_visit(node)

    # -- dunder / attribute escapes -------------------------------------
    def visit_Attribute(self, node):
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            self._fail(node, "access to dunder attribute %r is not allowed" % node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in FORBIDDEN_NAMES:
            self._fail(node, "use of %r is not allowed" % node.id)
        if node.id.startswith("__") and node.id.endswith("__"):
            self._fail(node, "use of dunder name %r is not allowed" % node.id)
        self.generic_visit(node)

    # -- misc dangerous constructs --------------------------------------
    def visit_Global(self, node):
        self._fail(node, "'global' is not allowed")

    def visit_Nonlocal(self, node):
        self._fail(node, "'nonlocal' is not allowed")


def validate(code):
    """Parse and statically validate ``code``. Raises :class:`ValidationError`.

    Returns the parsed AST on success (so the caller can compile it).
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValidationError("Syntax error: %s" % exc.msg)
    _Validator().visit(tree)
    return tree
