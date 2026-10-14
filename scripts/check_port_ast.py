#!/usr/bin/env python3
"""Static equivalence check between an original module and its English port.

The port is produced by a pure identifier rename (``docs/port_identifier_map.tsv``) followed by translation of the
prose (comments, docstrings, free-text messages). This script proves, without importing or executing either file,
that the two syntax trees are identical once:

* every identifier of the original (names, arguments, attributes, keywords, imports, and string literals that are
  identifier tokens) is renamed with the map;
* docstrings are removed from both sides;
* free-text string fragments (any string containing whitespace) are replaced by a placeholder on both sides.

String literals without whitespace (dictionary keys, format codes, tensor names, environment variable names) are
compared exactly after the rename. Any remaining difference is printed as a unified diff of ``ast.dump`` lines and
must be reviewed by hand (expected ones: import plumbing and configurable paths).

Usage::

    python3 scripts/check_port_ast.py --map docs/port_identifier_map.tsv ORIGINAL.py PORTED.py
"""
from __future__ import annotations

import argparse
import ast
import difflib
import os
import re
import sys

TOKEN = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_])")
TEXT = "<TEXT>"


def load_map(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or "\t" not in line:
                continue
            src, dst = line.split("\t", 1)
            out[src] = dst
    return out


class Normalize(ast.NodeTransformer):
    """Rename identifiers (when ``mapping`` is given), drop docstrings and blank out free text."""

    def __init__(self, mapping: dict | None, modules: dict | None = None, external: dict | None = None):
        self.m = mapping or {}
        self.mods = modules or {}
        ext = external or {}
        self.ext_modules = ext.get("module", set())
        self.ext_aliases = ext.get("alias", set())
        self.ext_keys = ext.get("key", set())

    def external_module(self, dotted: str) -> bool:
        return any(dotted == e or dotted.startswith(e + ".") for e in self.ext_modules)

    def name(self, s):
        return self.m.get(s, s) if s is not None else s

    def text(self, s: str) -> str:
        if any(c.isspace() for c in s):
            return TEXT
        if s in self.ext_keys:
            return s
        if s.startswith("/") and s.count("/") > 1:
            return "<ABSOLUTE-PATH>"          # machine paths of the original tree; the port has none
        return TOKEN.sub(lambda mo: self.m.get(mo.group(1), mo.group(1)), s)

    def _strip_doc(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return node

    def visit_Module(self, node):
        self._strip_doc(node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._strip_doc(node)
        node.name = self.name(node.name)
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._strip_doc(node)
        node.name = self.name(node.name)
        return self.generic_visit(node)

    def visit_Name(self, node):
        node.id = self.name(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.name(node.arg)
        node.annotation = self.visit(node.annotation) if node.annotation else None
        return node

    def visit_Attribute(self, node):
        if not (isinstance(node.value, ast.Name) and node.value.id in self.ext_aliases):
            node.attr = self.name(node.attr)
        return self.generic_visit(node)

    def visit_keyword(self, node):
        node.arg = self.name(node.arg)
        return self.generic_visit(node)

    def visit_Import(self, node):
        # ``import X as A`` of a ported module becomes ``from P import M as A`` in the port
        if len(node.names) == 1 and node.names[0].name in self.mods and node.names[0].asname:
            pkg, mod = self.mods[node.names[0].name].rsplit(".", 1)
            return ast.copy_location(ast.ImportFrom(module=pkg, names=[ast.alias(mod, self.name(node.names[0].asname))],
                                                    level=0), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        # ``__import__("X")`` of a ported module becomes ``__import__("P.M", fromlist=(<attribute>,))``
        node = self.generic_visit(node)
        if self.mods and isinstance(node.func, ast.Name) and node.func.id == "__import__" and len(node.args) == 1 \
                and isinstance(node.args[0], ast.Constant) and node.args[0].value in self.mods.values() \
                and not node.keywords:
            node.keywords = [ast.keyword("fromlist", ast.Tuple([ast.Constant("<ATTR>")], ast.Load()))]
        return node

    def visit_alias(self, node):
        if self.external_module(node.name):
            node.asname = self.name(node.asname)
            return node
        node.name = ".".join(self.name(p) for p in node.name.split("."))
        node.asname = self.name(node.asname)
        return node

    def visit_ImportFrom(self, node):
        if node.module in self.mods:
            node.module = self.mods[node.module]
            return self.generic_visit(node)
        if node.module and self.external_module(node.module):
            for a in node.names:
                a.asname = self.name(a.asname)
            return node
        if node.module:
            node.module = ".".join(self.name(p) for p in node.module.split("."))
        return self.generic_visit(node)

    def visit_Global(self, node):
        node.names = [self.name(n) for n in node.names]
        return node

    visit_Nonlocal = visit_Global

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = self.mods.get(node.value) or self.text(node.value)
        return node

    def visit_JoinedStr(self, node):
        node = self.generic_visit(node)
        # adjacent free-text fragments collapse to one placeholder, so splitting a sentence differently is not a diff
        vals = []
        for v in node.values:
            if isinstance(v, ast.Constant) and v.value == TEXT and vals and isinstance(vals[-1], ast.Constant) \
                    and vals[-1].value == TEXT:
                continue
            vals.append(v)
        node.values = vals
        return node


def dump_lines(tree: ast.AST) -> list:
    out = []
    for stmt in tree.body:
        out.extend(ast.dump(stmt, indent=1).splitlines())
    return out


def strip_fromlist_attr(tree: ast.AST) -> ast.AST:
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "__import__":
            for k in n.keywords:
                if k.arg == "fromlist":
                    k.value = ast.Tuple([ast.Constant("<ATTR>")], ast.Load())
    return tree


def load_external(path: str, ported: str) -> dict:
    """Entries of ``port_external_names.tsv`` that apply to the file name of ``ported``."""
    out: dict = {}
    base = ported.rsplit("/", 1)[-1]
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or line.count("\t") != 2:
                continue
            f, kind, name = line.rstrip("\n").split("\t")
            if f == base:
                out.setdefault(kind, set()).add(name)
    return out


def compare(original: str, ported: str, mapping: dict, modules: dict | None = None,
            external: dict | None = None) -> list:
    a = Normalize(mapping, modules, external).visit(ast.parse(open(original, encoding="utf-8").read()))
    b = strip_fromlist_attr(Normalize(None).visit(ast.parse(open(ported, encoding="utf-8").read())))
    return list(difflib.unified_diff(dump_lines(a), dump_lines(b), "original(renamed)", "port", lineterm="", n=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", required=True, help="identifier map, TSV: original<TAB>english")
    ap.add_argument("--modules", help="module map, TSV: original module<TAB>dotted English module")
    ap.add_argument("--external", help="names of unported code kept verbatim, TSV: file<TAB>module|alias|key<TAB>name")
    ap.add_argument("original")
    ap.add_argument("ported")
    ap.add_argument("--max-lines", type=int, default=400)
    ap.add_argument("--expected-dir", help="directory of reviewed plumbing diffs (<ported file name>.diff); a diff "
                                           "identical to the reviewed one counts as equivalent")
    ap.add_argument("--write-expected", action="store_true", help="store the current diff as the reviewed one")
    args = ap.parse_args()
    diff = compare(args.original, args.ported, load_map(args.map), load_map(args.modules) if args.modules else None,
                   load_external(args.external, args.ported) if args.external else None)
    if args.expected_dir:
        exp = os.path.join(args.expected_dir, os.path.basename(args.ported) + ".diff")
        if args.write_expected:
            if diff:
                os.makedirs(args.expected_dir, exist_ok=True)
                with open(exp, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(diff) + "\n")
            elif os.path.exists(exp):
                os.remove(exp)
        if diff and os.path.exists(exp) and open(exp, encoding="utf-8").read() == "\n".join(diff) + "\n":
            print(f"IDENTICAL  {args.ported}  (only the reviewed import/path plumbing differs)")
            return 0
    if not diff:
        print(f"IDENTICAL  {args.ported}")
        return 0
    print(f"DIFFERENT  {args.ported}  ({len(diff)} diff lines)")
    print("\n".join(diff[: args.max_lines]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
