"""Dependency annotation scheme helpers and embedded GRS runtime.

The bundled official UD/SUD GRS files are interpreted by the Python runtime
defined in this module. Compatibility modules re-export these definitions.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from io import StringIO
from itertools import chain
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unicodedata
from typing import Iterable

from .conllu import SentenceList, SentenceLists, TokenList, parse_incr
from .format_changer import read_conllu, write_conllu
from .tree_manipulation import convert_ud_function_dependents, restore_ud_function_dependents

SUD_RULES_REVISION = '7cd14fd1a5e106c18f2ddc9a597e02fe86fddb78'

"""GRS declarations and read-only matching; commands/strategies stay executor-owned.

parse_grs(path) returns GRS with qualified rules/packages/strategies dictionaries.
find_match(graph, rule) returns a variable -> node/edge ID dictionary or None.
Graph nodes are flat attribute dictionaries (feats/misc may also be dictionaries),
edges are dictionaries with src, dst, attrs, and order is a list of node IDs.
Unsupported syntax raises UnsupportedGRS instead of silently weakening a rule.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re
import unicodedata
from functools import lru_cache


class UnsupportedGRS(ValueError):
    pass


def _clean(text):
    return re.sub(r'"(?:\\.|[^"\\])*"|%[^\n]*',
                  lambda m: m[0] if m[0].startswith('"') else '', text)


@lru_cache(maxsize=4096)
def _split(text, delimiter=';'):
    parts, start, quoted, escaped, depth = [], 0, False, False, 0
    for i, c in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c in '[({':
            depth += 1
        elif c in '])}':
            depth -= 1
        elif c == delimiter and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return tuple(p for p in parts if p)


def _block(text, opening):
    depth, quoted, escaped = 1, False, False
    for i in range(opening + 1, len(text)):
        c = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if not depth:
                return text[opening + 1:i], i + 1
    raise UnsupportedGRS('Unclosed GRS block')


@dataclass
class Constraint:
    key: str
    op: str
    value: str = ''


def _constraints(text):
    result = []
    for item in _split(text, ','):
        m = re.fullmatch(r'([\w]+)\s*(<>|<=|>=|=|<|>)\s*(.+)', item, re.S)
        if m:
            result.append(Constraint(*m.groups()))
        elif re.fullmatch(r'!?\w+', item):
            result.append(Constraint(item.lstrip('!'), 'absent' if item.startswith('!') else 'present'))
        else:
            raise UnsupportedGRS('Unsupported feature constraint: ' + item)
    return result


@dataclass
class Pattern:
    nodes: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    comparisons: list = field(default_factory=list)


def parse_pattern(text):
    pattern = Pattern()
    for item in _split(_clean(text)):
        m = re.fullmatch(r'(?:(\w+)\s*:\s*)?(\w+|\*)\s*(?:-\[(.*?)\]\s*)?->\s*(\w+|\*)', item, re.S)
        if m:
            name, src, label, dst = m.groups()
            for node in (src, dst):
                if node != '*':
                    pattern.nodes.setdefault(node, [])
            structured = label is not None and bool(re.search(r'[=!,<>]', label))
            pattern.edges.append((name, src, dst, _constraints(label) if structured else label))
            continue
        m = re.fullmatch(r'(\w+)\s*\[(.*)\]', item, re.S)
        if m:
            pattern.nodes.setdefault(m[1], []).extend(_constraints(m[2]))
            continue
        m = re.fullmatch(r'(!?)(\w+\.\w+)', item)
        if m:
            pattern.comparisons.append((m[2], 'absent' if m[1] else 'present', ''))
            continue
        m = re.fullmatch(r'([\w.]+)\s*(<<|>>|><|<>|<=|>=|=|<|>)\s*(.+)', item, re.S)
        if m:
            pattern.comparisons.append(m.groups())
            continue
        raise UnsupportedGRS('Unsupported pattern statement: ' + item)
    edge_names = {edge[0] for edge in pattern.edges if edge[0]}
    for left, op, right in pattern.comparisons:
        if op in ('<<', '>>', '<', '>') and re.fullmatch(r'\w+', left) and re.fullmatch(r'\w+', right):
            for name in (left, right):
                if name not in edge_names:
                    pattern.nodes.setdefault(name, [])
        elif op in ('present', 'absent') and left.split('.')[0] not in edge_names:
            pattern.nodes.setdefault(left.split('.')[0], [])
    return pattern


@dataclass
class Rule:
    name: str
    pattern: Pattern
    without: list
    commands: str
    source: str = ''


@dataclass
class GRS:
    rules: dict = field(default_factory=dict)
    packages: dict = field(default_factory=dict)
    strategies: dict = field(default_factory=dict)
    includes: list = field(default_factory=list)

    @property
    def strats(self):
        return self.strategies


def parse_grs(path):
    """Load relative includes; direct package rules follow Grew loader order.

    Grew unfolds each declaration list with a left fold that prepends ordinary
    declarations and prepends recursively unfolded includes. Thus each package
    runs in reverse flattened source order, independently of nested packages.
    Rule/strategy dictionaries are lookups, not execution-order lists.
    """
    grs, visited = GRS(), set()

    def declarations(text, source, scope=''):
        pos = 0
        while pos < len(text):
            m = re.match(r'\s*;?\s*', text[pos:])
            pos += m.end()
            if pos == len(text):
                break
            inc = re.match(r'include\s+"([^"]+)"', text[pos:])
            if inc:
                target = (source.parent / inc[1]).resolve()
                grs.includes.append(str(target))
                load(target, scope)
                pos += inc.end()
                continue
            header = re.match(r'(package|rule|strat)\s+([\w-]+)\s*\{', text[pos:])
            if not header:
                raise UnsupportedGRS(f'Unsupported declaration in {source}: {text[pos:pos+100]}')
            kind, name = header.groups()
            body, pos = _block(text, pos + header.end() - 1)
            qualified = scope + name
            if kind == 'package':
                grs.packages[qualified] = []
                declarations(body, source, qualified + '.')
            elif kind == 'strat':
                grs.strategies[qualified] = body.strip()
            else:
                sections, offset = {}, 0
                while offset < len(body):
                    h = re.match(r'\s*(pattern|without|commands)\s*\{', body[offset:])
                    if not h:
                        if not body[offset:].strip():
                            break
                        raise UnsupportedGRS('Unsupported rule section: ' + body[offset:offset+80])
                    content, offset = _block(body, offset + h.end() - 1)
                    sections.setdefault(h[1], []).append(content)
                if len(sections.get('pattern', [])) != 1:
                    raise UnsupportedGRS('Rule requires one pattern: ' + qualified)
                grs.rules[qualified] = Rule(qualified, parse_pattern(sections['pattern'][0]),
                    [parse_pattern(p) for p in sections.get('without', [])],
                    ';'.join(sections.get('commands', [])).strip(), str(source))
                if scope:
                    grs.packages[scope[:-1]].append(qualified)

    def load(source, scope=''):
        key = (source, scope)
        if key in visited:
            return
        visited.add(key)
        declarations(_clean(source.read_text(encoding='utf-8-sig')), source, scope)

    load(Path(path).resolve())
    # Reversing the flattened direct rules also reverses each include's rules,
    # reproducing unfold_grs's unfolded_sub @ acc rather than reversing twice.
    for members in grs.packages.values():
        members.reverse()
    return grs


def label_attrs(label):
    """Decode CoNLL-U relation base:subtype@deep into GRS attributes."""
    base, sep, deep = str(label).partition('@')
    first, colon, second = base.partition(':')
    result = {'1': first}
    if colon:
        result['2'] = second
    if sep:
        result['deep'] = deep
    return result


def edge_attrs(edge):
    attrs = dict(edge.get('attrs', {}))
    label = attrs.get('label', edge.get('label'))
    if label is not None:
        attrs = {**label_attrs(label), **attrs}
    attrs['label'] = str(attrs.get('1', '')) + (':' + str(attrs['2']) if '2' in attrs else '') + ('@' + str(attrs['deep']) if 'deep' in attrs else '')
    return attrs


def _atomic_label_matches(attrs, choices):
    """Grew Label_cst.Pos/Neg compare the complete edge feature structure."""
    features = {key: value for key, value in attrs.items() if key != 'label'}
    return any(features == label_attrs(choice) for choice in choices)


def node_attrs(node):
    attrs = dict(node)
    for key in ('feats', 'misc'):
        values = attrs.get(key, {})
        if isinstance(values, dict):
            attrs.update(values)
        elif isinstance(values, str) and values != '_':
            attrs.update(piece.split('=', 1) for piece in values.split('|') if '=' in piece)
    if 'form' in attrs:
        attrs.setdefault('wordform', attrs['form'])
    return attrs


_MISSING = object()


@lru_cache(maxsize=4096)
def _references(value):
    return tuple(re.findall(r'\b(\w+)\.\w+\b', value))


@lru_cache(maxsize=4096)
def _quoted_literal(value):
    """Decode string_lex in grew_lexer.mll, without regex or NFC processing."""
    if not re.fullmatch(r'"(?:\\.|[^"\\])*"', value, re.S):
        raise UnsupportedGRS('Invalid quoted literal: ' + value)
    result, escaped = [], False
    for char in value[1:-1]:
        if char == '\\':
            if escaped:
                result.append('\\')
            escaped = not escaped
        elif char == '\n':
            result.append(char)
        else:
            if escaped and char != '"':
                result.append('\\')
            result.append(char)
            escaped = False
    return ''.join(result)


@lru_cache(maxsize=4096)
def _literal(value):
    return unicodedata.normalize('NFC', _quoted_literal(value))


@lru_cache(maxsize=4096)
def _regex_literal(value):
    """Decode Grew string_lex escapes, then translate Str regex operators.

    grew_lexer.mll collapses only escaped quotes and backslashes; other
    escapes survive. grew_ast.ml compiles re strings with Str.regexp, whose
    alternation and grouping are escaped, unlike Python's regex syntax.
    """
    raw = _quoted_literal(value)
    result, index, in_class = [], 0, False
    while index < len(raw):
        char = raw[index]
        if char == '\\' and not in_class and index + 1 < len(raw):
            following = raw[index + 1]
            if following in '|()':
                result.append(following)
            elif following in '123456789b':
                result.append('\\' + following)
            else:
                result.append(re.escape(following))
            index += 2
            continue
        if char == '[' and not in_class:
            in_class = True
        elif char == ']' and in_class:
            in_class = False
        elif not in_class and char in '|(){}':
            char = '\\' + char
        result.append(char)
        index += 1
    return ''.join(result)


def _resolve(value, graph, bindings, attribute_cache=None):
    value = value.strip()
    if value.startswith('"'):
        return _literal(value)
    if re.fullmatch(r'\w+\.\w+', value):
        var, key = value.split('.', 1)
        if var not in bindings:
            return _MISSING
        ident = bindings[var]
        # Variable namespaces are disjoint within a pattern; callers supply kind.
        kind = bindings.get('__kinds__', {}).get(var, 'node')
        if attribute_cache is not None:
            attrs = attribute_cache[kind](ident)
        else:
            attrs = edge_attrs(graph.edges[ident]) if kind == 'edge' else node_attrs(graph.nodes[ident])
        return attrs.get(key, _MISSING)
    return value


def _compare(actual, op, expected, graph, bindings, attribute_cache=None):
    if op == 'present':
        return actual is not _MISSING
    if op == 'absent':
        return actual is _MISSING
    if expected == '*':
        return (actual is not _MISSING) if op == '=' else (actual is _MISSING)
    if actual is _MISSING:
        return False
    if expected.startswith('re"'):
        hit = re.fullmatch(_regex_literal(expected[2:]), str(actual)) is not None
        return hit if op == '=' else not hit
    values = [_resolve(v, graph, bindings, attribute_cache) for v in _split(expected, '|')]
    if any(v is _MISSING for v in values):
        return False
    if op in ('=', '<>'):
        hit = str(actual) in [str(v) for v in values]
        return hit if op == '=' else not hit
    try:
        a, b = float(actual), float(values[0])
    except (TypeError, ValueError):
        a, b = str(actual), str(values[0])
    return {'<': a < b, '>': a > b, '<=': a <= b, '>=': a >= b}[op]


def iter_pattern_matches(graph, pattern, bindings=None):
    """Yield matches in Grew root/PID and edge-following priority order.

    GIDs must be comparable, as in Grew's integer-ID graph. Token order only
    controls precedence constraints, not candidate priority.
    """
    bindings = dict(bindings or {})
    kinds = dict(bindings.get('__kinds__', {}))
    # A standalone feature in a without clause may refer to an inherited edge.
    if any(kinds.get(name) == 'edge' for name in pattern.nodes):
        pattern = Pattern({name: tests for name, tests in pattern.nodes.items()
                           if kinds.get(name) != 'edge'}, pattern.edges, pattern.comparisons)
    kinds.update({n: 'node' for n in pattern.nodes})
    kinds.update({e[0]: 'edge' for e in pattern.edges if e[0]})
    bindings['__kinds__'] = kinds
    positions = {n: i for i, n in enumerate(graph.order)}
    # These caches live only while matching this immutable graph snapshot.
    node_cache, edge_cache = {}, {}

    def get_node(nid):
        if nid not in node_cache:
            node_cache[nid] = node_attrs(graph.nodes[nid])
        return node_cache[nid]

    def get_edge(eid):
        if eid not in edge_cache:
            edge_cache[eid] = edge_attrs(graph.edges[eid])
        return edge_cache[eid]

    attribute_cache = {'node': get_node, 'edge': get_edge}

    def constraints(attrs, tests, env):
        return all(_compare(attrs.get(c.key, _MISSING), c.op, c.value, graph, env, attribute_cache) for c in tests)

    def finish(env):
        def span(name):
            edge = graph.edges[env[name]]
            endpoints = [positions.get(edge[key]) for key in ('src', 'dst')]
            return None if None in endpoints else sorted(endpoints)

        for left, op, right in pattern.comparisons:
            right = right.strip()
            if op == '><':
                a, b = span(left), span(right)
                if a is None or b is None or not (a[0] < b[0] < a[1] < b[1] or b[0] < a[0] < b[1] < a[1]):
                    return
            elif op in ('<<', '>>', '<', '>') and '.' not in left and re.fullmatch(r'\w+', right):
                if op in ('>>', '>'):
                    left, right = right, left
                if kinds.get(right) == 'edge':
                    b = span(right)
                    if kinds.get(left) == 'edge':
                        a = span(left)
                        if a is None or b is None or not (b[0] <= a[0] and a[1] <= b[1] and a[0] < b[1] and b[0] < a[1]):
                            return
                    else:
                        a = positions.get(env.get(left))
                        if a is None or b is None or not b[0] < a < b[1]:
                            return
                else:
                    a, b = positions.get(env.get(left)), positions.get(env.get(right))
                    if a is None or b is None or not (b == a + 1 if op in ('<', '>') else a < b):
                        return
            elif not _compare(_resolve(left, graph, env, attribute_cache), op, right, graph, env, attribute_cache):
                return
        # Recheck after all bindings exist, including cross-variable constraints.
        if not all(constraints(get_node(env[n]), tests, env) for n, tests in pattern.nodes.items()):
            return
        for name, src, dst, tests in pattern.edges:
            if name and isinstance(tests, list) and not constraints(get_edge(env[name]), tests, env):
                return
        yield env

    def pattern_edge_key(index):
        name, src, dst, tests = pattern.edges[index]
        label = tuple((c.key, c.op, c.value) for c in tests) if isinstance(tests, list) else ((tests or ''),)
        return dst, name is not None, name or '', repr(label)

    outgoing = {n: [] for n in pattern.nodes}
    anonymous = []
    for index, (_, src, dst, _) in enumerate(pattern.edges):
        if src == '*' or dst == '*':
            anonymous.append(index)
        else:
            outgoing[src].append(index)
    for indices in outgoing.values():
        indices.sort(key=pattern_edge_key, reverse=True)

    targets = {dst for _, src, dst, _ in pattern.edges if src != '*' and dst != '*'}
    names = sorted(pattern.nodes, reverse=True)
    # Grew's stable sort puts roots first, retaining descending alphabetical PID.
    names.sort(key=lambda n: n in targets)
    remaining = tuple(n for n in names if n not in bindings)
    initial_edges = tuple(index for n in sorted(pattern.nodes, reverse=True)
                          if n in bindings for index in outgoing[n])

    def graph_edge_key(item):
        eid, edge = item
        attrs = get_edge(eid)
        # G_edge is a sorted feature list; label is a computed convenience field.
        features = tuple(sorted((str(k), str(v)) for k, v in attrs.items() if k != 'label'))
        return edge['dst'], features, eid

    def edge_fits(index, eid, edge, env):
        name, src, dst, tests = pattern.edges[index]
        if name and name in env and env[name] != eid:
            return False
        if src != '*' and src in env and edge['src'] != env[src]:
            return False
        if dst != '*' and dst in env and edge['dst'] != env[dst]:
            return False
        if eid not in static_edges[index]:
            return False
        attrs = get_edge(eid)
        if isinstance(tests, list):
            ready = [c for c, refs in dynamic_edge_tests[index] if all(v in env for v in refs)]
            return constraints(attrs, ready, env)
        if tests is None:
            return True
        if not dynamic_edge_tests[index]:
            return True
        hit = _atomic_label_matches(attrs, [_resolve(v, graph, env, attribute_cache)
                                           for v in _split(tests.lstrip('^'), '|')])
        return hit != tests.startswith('^')

    def bind_node(name, nid, env, todo_nodes, todo_edges):
        if nid not in node_candidates[name]:
            return
        used = {env[n] for n in env if kinds.get(n) == 'node' and n != name}
        if nid not in graph.nodes or nid in used:
            return
        next_env = dict(env)
        next_env[name] = nid
        local = local_node_tests[name]
        if constraints(get_node(nid), local, next_env):
            yield from extend(next_env, tuple(n for n in todo_nodes if n != name),
                              tuple(outgoing[name]) + todo_edges)

    def extend(env, todo_nodes, todo_edges):
        if todo_edges:
            index, *tail = todo_edges
            name, src, dst, tests = pattern.edges[index]
            # New targets reverse the adjacency fold; bound targets use the
            # ascending label list returned by P_edge.match_list.
            unbound = dst != '*' and dst not in env
            candidates = ordered_outgoing.get(env.get(src), ()) if src != '*' else ordered_edges
            if unbound:
                candidates = reversed(candidates)
            passed = False
            for eid, edge in candidates:
                if name and name in env and env[name] != eid:
                    continue
                next_env = dict(env)
                if name:
                    next_env[name] = eid
                if not edge_fits(index, eid, edge, next_env):
                    continue
                if not name and not unbound:
                    if passed:
                        continue
                    passed = True
                if unbound:
                    yield from bind_node(dst, edge['dst'], next_env, todo_nodes, tuple(tail))
                else:
                    yield from extend(next_env, todo_nodes, tuple(tail))
            return
        if todo_nodes:
            name = todo_nodes[0]
            for nid in node_candidates_ordered[name]:
                yield from bind_node(name, nid, env, todo_nodes, ())
            return
        if anonymous:
            yield from check_anonymous(0, env)
        else:
            yield from finish(env)

    def check_anonymous(index, env):
        if index == len(anonymous):
            yield from finish(env)
            return
        edge_index = anonymous[index]
        name = pattern.edges[edge_index][0]
        src = pattern.edges[edge_index][1]
        candidates = ordered_outgoing.get(env.get(src), ()) if src != '*' else ordered_edges
        for eid, edge in reversed(candidates):
            if edge_fits(edge_index, eid, edge, env):
                next_env = dict(env)
                if name:
                    next_env[name] = eid
                yield from check_anonymous(index + 1, next_env)
                if not name:
                    break

    local_node_tests = {name: tuple(c for c in tests if not _references(c.value))
                        for name, tests in pattern.nodes.items()}
    # Static filters eliminate impossible branches without changing traversal.
    static_edges, dynamic_edge_tests = {}, {}
    for index, (_, src, dst, tests) in enumerate(pattern.edges):
        if isinstance(tests, list):
            static = tuple(c for c in tests if not _references(c.value))
            dynamic_edge_tests[index] = tuple((c, _references(c.value)) for c in tests if _references(c.value))
            static_edges[index] = {eid for eid in graph.edges
                                   if constraints(get_edge(eid), static, bindings)}
        elif tests is None or _references(tests):
            static_edges[index] = set(graph.edges)
            dynamic_edge_tests[index] = bool(tests is not None)
        else:
            labels = tuple(_resolve(v, graph, bindings, attribute_cache)
                           for v in _split(tests.lstrip('^'), '|'))
            static_edges[index] = {eid for eid in graph.edges
                                   if _atomic_label_matches(get_edge(eid), labels) != tests.startswith('^')}
            dynamic_edge_tests[index] = False
        # An empty static edge domain proves this entire pattern impossible.
        if not static_edges[index]:
            return
    node_candidates = {name: ({bindings[name]} & graph.nodes.keys()
                             if name in bindings else set(graph.nodes))
                       for name in pattern.nodes}
    for index, (_, src, dst, _) in enumerate(pattern.edges):
        if src != '*':
            node_candidates[src].intersection_update(graph.edges[eid]['src'] for eid in static_edges[index])
        if dst != '*':
            node_candidates[dst].intersection_update(graph.edges[eid]['dst'] for eid in static_edges[index])
    for name, candidates in node_candidates.items():
        tests = local_node_tests[name]
        if tests:
            candidates.intersection_update(nid for nid in tuple(candidates)
                                           if constraints(get_node(nid), tests, bindings))
        if not candidates:
            return
    # Sorting is only needed for a feasible pattern, and retains Grew priority.
    ordered_nodes = tuple(sorted(graph.nodes, reverse=True))
    ordered_edges = tuple(sorted(graph.edges.items(), key=graph_edge_key))
    ordered_outgoing = {}
    for item in ordered_edges:
        ordered_outgoing.setdefault(item[1]['src'], []).append(item)
    node_candidates_ordered = {name: tuple(nid for nid in ordered_nodes if nid in candidates)
                               for name, candidates in node_candidates.items()}
    yield from extend(bindings, remaining, initial_edges)


def iter_matches(graph, rule):
    for env in iter_pattern_matches(graph, rule.pattern):
        if not any(next(iter_pattern_matches(graph, p, env), None) is not None for p in rule.without):
            yield {k: v for k, v in env.items() if k != '__kinds__'}


def find_match(graph, rule):
    """Return the first match, or None; never mutate the graph."""
    return next(iter_matches(graph, rule), None)



"""Command execution and deterministic strategies for official UD/SUD GRS files.

Integration: execute_rules(graph, parse_grs(path), 'UD_to_SUD_main').
Reverse conversion uses the same executor with 'SUD_to_UD_main'. In particular,
flatten_idioms uses Iter(one), where one is an ordinary scoped strategy reference.
Graphs expose nodes/edges dictionaries and an order list (or dictionary keys).
Matcher callback: find_match(graph, rule) -> flat variable-to-ID dict or None.
Nested {'nodes': ..., 'edges': ...} bindings are also accepted. Edge labels are
attribute structures with string keys '1', '2', 'deep'; label copies copy ALL
edge attributes, including temporary REV/level fields. No package source edits
or graph methods are required. Optional add_node/add_edge methods allocate IDs.

This is deterministic execution: Iter and Pick choose the first result, rather
than enumerating all Grew normal forms. Unsupported syntax fails explicitly.
"""
from copy import copy, deepcopy
from functools import lru_cache
import re

try:
    from .grs_matcher import UnsupportedGRS, _literal, _regex_literal
except ImportError:
    from grs_matcher import UnsupportedGRS, _literal, _regex_literal


class GRSExecutionError(ValueError):
    """Unsupported syntax, invalid command, or rewrite limit reached."""


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _split(text, delimiter=';'):
    result, start, depth, quote, escaped = [], 0, 0, False, False
    for i, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quote = False
        elif char == '"':
            quote = True
        elif char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == delimiter and depth == 0:
            result.append(text[start:i].strip())
            start = i + 1
    if quote or depth:
        raise GRSExecutionError('Unbalanced expression: ' + text)
    result.append(text[start:].strip())
    return [item for item in result if item]


def _clean(text):
    return re.sub(r'"(?:\\.|[^"\\])*"|%[^\n]*',
                  lambda m: m[0] if m[0].startswith('"') else '', text)


def _label(text):
    base, at, deep = str(text).partition('@')
    first, colon, second = base.partition(':')
    attrs = {'1': first}
    if colon:
        attrs['2'] = second
    if at:
        attrs['deep'] = deep
    return attrs


def _attrs(edge):
    attrs = edge.setdefault('attrs', {})
    label = attrs.pop('label', edge.pop('label', None))
    if label is not None and '1' not in attrs:
        attrs.update(_label(label))
    return attrs


def _fresh(mapping, prefix):
    i = 1
    while prefix + str(i) in mapping:
        i += 1
    return prefix + str(i)


def _commit(target, source):
    if isinstance(target, dict):
        target.clear()
        target.update(source)
    elif hasattr(target, '__dict__'):
        target.__dict__.clear()
        target.__dict__.update(source.__dict__)
    else:
        for key in ('nodes', 'edges', 'order'):
            setattr(target, key, _get(source, key))


def _clone_graph(graph):
    """Copy rewrite state, sharing read-only sentence/row conversion context.

    Graph allocation methods must mutate only nodes, edges, order, or scalar
    instance counters. Other graph attributes are execution context, not state.
    """
    candidate = copy(graph)
    memo = {}
    for key in ('nodes', 'edges', 'order'):
        cloned = deepcopy(_get(graph, key), memo)
        if isinstance(candidate, dict):
            candidate[key] = cloned
        else:
            setattr(candidate, key, cloned)
    return candidate


@lru_cache(maxsize=4096)
def _command_block(commands):
    return tuple(_split(_clean(commands)))


def _execute(graph, commands, match):
    nodes, edges, order = (_get(graph, key) for key in ('nodes', 'edges', 'order'))
    if nodes is None or edges is None or order is None:
        raise GRSExecutionError('Graph requires nodes, edges, and order')
    if 'nodes' in match and isinstance(match['nodes'], dict):
        node_bindings, edge_bindings = dict(match['nodes']), dict(match.get('edges', {}))
    else:
        # Matcher kinds disambiguate IDs shared by node and edge namespaces.
        kinds = match.get('__kinds__', {})
        node_bindings, edge_bindings = {}, {}
        for name, identifier in match.items():
            if name == '__kinds__':
                continue
            kind = kinds.get(name)
            if kind is not None:
                if kind not in ('node', 'edge'):
                    raise GRSExecutionError('Unknown binding kind: ' + str(kind))
                mapping = nodes if kind == 'node' else edges
                if identifier not in mapping:
                    raise GRSExecutionError('Unknown ' + kind + ' binding: ' + name)
                (node_bindings if kind == 'node' else edge_bindings)[name] = identifier
            else:
                if identifier in nodes:
                    node_bindings[name] = identifier
                if identifier in edges:
                    edge_bindings[name] = identifier
        ambiguous = node_bindings.keys() & edge_bindings.keys()
        if ambiguous:
            raise GRSExecutionError('Ambiguous node/edge IDs; use nested bindings: ' + str(ambiguous))
    local_nodes = set(node_bindings.values())

    def node(name):
        if name not in node_bindings or node_bindings[name] not in nodes:
            raise GRSExecutionError('Unknown node: ' + name)
        return node_bindings[name]

    def attrs(name):
        if name in edge_bindings:
            if edge_bindings[name] not in edges:
                raise GRSExecutionError('Deleted edge: ' + name)
            return _attrs(edges[edge_bindings[name]])
        return nodes[node(name)]

    def value(expr):
        parts = _split(expr.strip(), '+')
        if len(parts) > 1:
            return ''.join(str(value(part)) for part in parts)
        expr = expr.strip()
        if expr.startswith('"'):
            try:
                return _literal(expr)
            except UnsupportedGRS as exc:
                raise GRSExecutionError('Invalid quoted value: ' + expr) from exc
        m = re.fullmatch(r'(\w+)\.(\w+)(?:\[(-?\d*)?:(-?\d*)?\])?', expr)
        if m:
            source = attrs(m[1])
            if m[2] == 'label' and m[1] in edge_bindings:
                return deepcopy(source)
            if m[2] not in source:
                raise GRSExecutionError('Missing feature: ' + expr)
            result = deepcopy(source[m[2]])
            if '[' in expr:
                result = str(result)[slice(int(m[3]) if m[3] else None,
                                          int(m[4]) if m[4] else None)]
            return result
        if re.fullmatch(r'[\w:@/.-]+', expr):
            return expr
        raise GRSExecutionError('Unsupported value expression: ' + expr)

    def edge_label(text):
        if text is None:
            return {}
        if '=' not in text:
            return _label(value(text))
        result = {}
        for item in _split(text, ','):
            m = re.fullmatch(r'(\w+)\s*=\s*(.+)', item, re.S)
            if not m:
                raise GRSExecutionError('Invalid edge attributes: ' + text)
            result[m[1]] = value(m[2])
        return result

    def selected(data, spec):
        if spec is None or spec.strip() == '*':
            return True
        spec = spec.strip()
        if not re.search(r'[=!,<>]', spec):
            negate = spec.startswith('^')
            labels = spec.lstrip('^').split('|')
            found = any(data == _label(label) for label in labels)
            return not found if negate else found
        for item in _split(spec, ','):
            m = re.fullmatch(r'(\w+)\s*(<>|=)\s*(.+)', item)
            if m:
                key, op, choices = m.groups()
                found = key in data and (choices == '*' or str(data[key]) in choices.split('|'))
                if (op == '=' and not found) or (op == '<>' and found):
                    return False
            elif re.fullmatch(r'!?\w+', item):
                if (item.lstrip('!') in data) == item.startswith('!'):
                    return False
            else:
                raise GRSExecutionError('Unsupported shift filter: ' + item)
        return True

    def place(identifier, anchor, side):
        if identifier in order:
            order.remove(identifier)
        if anchor not in order:
            raise GRSExecutionError('Unordered insertion anchor')
        order.insert(order.index(anchor) + (side == '>'), identifier)

    for command in _command_block(commands) if isinstance(commands, str) else commands:
        command = command.strip()
        m = re.fullmatch(r'(\w+)\.(\w+)\s*=\s*(.+)', command, re.S)
        if m:
            name, key, expression = m.groups()
            new_value = value(expression)
            destination = attrs(name)
            if key == 'label' and name in edge_bindings:
                destination.clear()
                destination.update(new_value if isinstance(new_value, dict) else _label(new_value))
            else:
                destination[key] = new_value
            continue
        m = re.fullmatch(r'del_feat\s+(\w+)\.(\w+)', command)
        if m:
            attrs(m[1]).pop(m[2], None)
            continue
        m = re.fullmatch(r'del_node\s+(\w+)', command)
        if m:
            identifier = node(m[1])
            del nodes[identifier]
            if identifier in order:
                order.remove(identifier)
            for eid, edge in list(edges.items()):
                if identifier in (edge['src'], edge['dst']):
                    del edges[eid]
            continue
        m = re.fullmatch(r'(add_node|insert)\s+(\w+)(?:\s*:\s*([<>])\s*(\w+))?', command)
        if m:
            kind, name, side, anchor_name = m.groups()
            anchor = node(anchor_name) if anchor_name else None
            if kind == 'add_node':
                if name in node_bindings or name in edge_bindings:
                    raise GRSExecutionError('Non-fresh variable: ' + name)
                method = _get(graph, 'add_node')
                identifier = method(before=anchor if side == '<' else None) if callable(method) else _fresh(nodes, '__grs_n')
                nodes.setdefault(identifier, {})
                node_bindings[name] = identifier
                local_nodes.add(identifier)
            else:
                if anchor is None:
                    raise GRSExecutionError('insert requires anchor')
                identifier = node(name)
            if anchor is not None:
                place(identifier, anchor, side)
            elif identifier in order:
                order.remove(identifier)
            continue
        m = re.fullmatch(r'unorder\s+(\w+)', command)
        if m:
            identifier = node(m[1])
            if identifier in order:
                order.remove(identifier)
            continue
        m = re.fullmatch(r'(add_edge|del_edge)\s+(?:(\w+)\s*:\s*)?(\w+)\s*(?:-\[(.*?)\]\s*)?->\s*(\w+)', command, re.S)
        if m:
            kind, name, src, label, dst = m.groups()
            src, dst, data = node(src), node(dst), edge_label(label)
            if kind == 'del_edge':
                found = [eid for eid, edge in edges.items() if edge['src'] == src and edge['dst'] == dst and _attrs(edge) == data]
                if not found:
                    raise GRSExecutionError('Missing edge: ' + command)
                for eid in found:
                    del edges[eid]
            else:
                if name and (name in node_bindings or name in edge_bindings):
                    raise GRSExecutionError('Non-fresh variable: ' + name)
                method = _get(graph, 'add_edge')
                eid = method(src, dst, deepcopy(data)) if callable(method) else _fresh(edges, '__grs_e')
                if not callable(method):
                    edges[eid] = {'src': src, 'dst': dst, 'attrs': data}
                if name:
                    edge_bindings[name] = eid
            continue
        m = re.fullmatch(r'del_edge\s+(\w+)', command)
        if m:
            name = m[1]
            if name not in edge_bindings or edge_bindings[name] not in edges:
                raise GRSExecutionError('Unknown edge: ' + name)
            del edges[edge_bindings[name]]
            continue
        m = re.fullmatch(r'(shift_in|shift_out|shift)\s+(\w+)\s*=(?:\[(.*?)\])?=>\s*(\w+)', command, re.S)
        if m:
            kind, src_name, spec, dst_name = m.groups()
            src, dst = node(src_name), node(dst_name)
            for edge in list(edges.values()):
                if not selected(_attrs(edge), spec):
                    continue
                if kind != 'shift_out' and edge['dst'] == src and edge['src'] not in local_nodes:
                    edge['dst'] = dst
                if kind != 'shift_in' and edge['src'] == src and edge['dst'] not in local_nodes:
                    edge['src'] = dst
            continue
        m = re.fullmatch(r'(append_feats|prepend_feats)\s+(?:((?:"(?:\\.|[^"\\])*")|\S+)\s+)?(\w+)\s*=(?:\[(.*?)\])?=>\s*(\w+)', command, re.S)
        if m:
            kind, separator, src_name, spec, dst_name = m.groups()
            source, destination = attrs(src_name), attrs(dst_name)
            sep = str(value(separator)) if separator else ''
            regex = None
            if spec is not None:
                rm = re.fullmatch(r're"(.*)"', spec.strip(), re.S)
                if not rm:
                    raise GRSExecutionError('Unsupported feature filter: ' + spec)
                regex = re.compile(_regex_literal(spec.strip()[2:]))
            for key, item in list(source.items()):
                if key in ('form', 'lemma', 'upos', 'xpos') or (regex and not regex.fullmatch(key)):
                    continue
                if key in destination:
                    left, right = (destination[key], item) if kind == 'append_feats' else (item, destination[key])
                    destination[key] = str(left) + sep + str(right)
                else:
                    destination[key] = deepcopy(item)
            continue
        raise GRSExecutionError('Unsupported command: ' + command)
    return graph


def execute_commands(graph, commands, match):
    """Apply a command block atomically; preserve input on command failure."""
    candidate = _clone_graph(graph)
    _execute(candidate, commands, match)
    _commit(graph, candidate)
    return graph


@lru_cache(maxsize=4096)
def parse_strategy(expression):
    """Return (operator, children) or ('Ref', name), checking all syntax."""
    expression = _clean(expression).strip()
    if re.fullmatch(r'[\w.-]+', expression):
        return ('Seq', ()) if expression == 'Empty' else ('Ref', expression)
    m = re.fullmatch(r'(Seq|Onf|Iter|Try|Pick|Alt)\s*\((.*)\)', expression, re.S)
    if not m:
        raise GRSExecutionError('Unsupported strategy: ' + expression)
    op, body = m.groups()
    children = tuple(parse_strategy(part) for part in _split(body, ','))
    if op not in ('Seq', 'Alt') and len(children) != 1:
        raise GRSExecutionError(op + ' requires exactly one argument')
    return op, children


class StrategyEvaluator:
    """First-match strategy evaluator with bounded, transactional evaluation.

    evaluate returns [] on failure or [graph] on success, without input mutation.
    run commits a successful result to its input. Iter deliberately uses the
    same deterministic path as Onf; it does not enumerate alternate matches.
    """

    def __init__(self, rules, packages, strategies, find_match, max_steps=10000, trace=None):
        self.rules, self.packages, self.strategies = rules, packages, strategies
        self.find_match, self.max_steps = find_match, max_steps
        self.steps = 0
        self.trace = trace

    def _resolve(self, name, scope):
        while True:
            qualified = scope + '.' + name if scope else name
            if qualified in self.rules or qualified in self.packages or qualified in self.strategies:
                return qualified
            if not scope:
                raise GRSExecutionError('Unknown strategy reference: ' + name)
            scope = scope.rpartition('.')[0]

    def _tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise GRSExecutionError('Rewrite step limit exceeded: ' + str(self.max_steps))

    def _rule(self, graph, name):
        rule = self.rules[name]
        match = self.find_match(graph, rule)
        if match is None:
            return None
        candidate = _clone_graph(graph)
        try:
            _execute(candidate, _get(rule, 'commands'), match)
        except GRSExecutionError as exc:
            raise GRSExecutionError(name + ': ' + str(exc)) from exc
        if all(_get(candidate, key) == _get(graph, key) for key in ('nodes', 'edges', 'order')):
            return None
        self._tick()
        if self.trace is not None:
            self.trace.append({'step': self.steps, 'rule': name,
                               'bindings': deepcopy(match)})
        return candidate

    def _eval(self, graph, expression, scope='', stack=()):
        op, args = expression
        if op == 'Ref':
            name = self._resolve(args, scope)
            if name in self.rules:
                return self._rule(graph, name)
            if name in self.packages:
                for reference in self.packages[name]:
                    rule_name = reference if isinstance(reference, str) else _get(reference, 'name')
                    # Only direct rules belong to a package application.
                    if rule_name.rpartition('.')[0] != name:
                        continue
                    result = self._rule(graph, rule_name)
                    if result is not None:
                        return result
                return None
            if name in stack:
                raise GRSExecutionError('Recursive strategy reference: ' + name)
            body = self.strategies[name]
            if not isinstance(body, str):
                body = _get(body, 'body', _get(body, 'expression'))
            return self._eval(graph, parse_strategy(body), name.rpartition('.')[0], stack + (name,))
        if op == 'Seq':
            current = graph
            for child in args:
                current = self._eval(current, child, scope, stack)
                if current is None:
                    return None
            return current
        if op == 'Alt':
            for child in args:
                result = self._eval(graph, child, scope, stack)
                if result is not None:
                    return result
            return None
        if op in ('Pick', 'Try'):
            result = self._eval(graph, args[0], scope, stack)
            return graph if result is None and op == 'Try' else result
        if op in ('Onf', 'Iter'):
            current = graph
            while True:
                result = self._eval(current, args[0], scope, stack)
                if result is None:
                    return current
                if all(_get(result, key) == _get(current, key) for key in ('nodes', 'edges', 'order')):
                    raise GRSExecutionError(op + ' operand succeeds without changing graph')
                current = result
        raise GRSExecutionError('Unsupported strategy operator: ' + op)

    def evaluate(self, graph, strategy):
        self.steps = 0
        try:
            result = self._eval(_clone_graph(graph), parse_strategy(strategy))
        except RecursionError as exc:
            raise GRSExecutionError('Strategy recursion limit exceeded') from exc
        return [] if result is None else [result]

    def run(self, graph, strategy):
        results = self.evaluate(graph, strategy)
        if not results:
            raise GRSExecutionError('Strategy is not productive: ' + strategy)
        _commit(graph, results[0])
        return graph


def execute_rules(graph, program, strategy='UD_to_SUD_main', max_steps=10000, matcher=None,
                  trace=None):
    """Execute a parsed GRS in place; return graph; fail without partial edits.

    Pass a list as trace to append {'step', 'rule', 'bindings'} for successful
    rewrites. It records attempted evaluation, including branches subsequently
    rolled back and progress before an exception; graph commit remains atomic.
    Trace is opt-in, bounded by max_steps, and never stores graph snapshots.
    """
    if matcher is None:
        matcher = find_match
    evaluator = StrategyEvaluator(_get(program, 'rules'), _get(program, 'packages'),
                                  _get(program, 'strategies'), matcher, max_steps, trace)
    return evaluator.run(graph, strategy)



"""Python execution of the pinned official UD/SUD graph rules."""

from copy import deepcopy
from pathlib import Path
from functools import lru_cache
import os

SUD_RULES_REVISION = '7cd14fd1a5e106c18f2ddc9a597e02fe86fddb78'

# Official UD/SUD MISC tables use these external names for GRS identifiers.
_MISC_NAMES = {
    **{f'{name}[{kind}]': f'{name}__{kind}'
       for name in ('Gender', 'Number', 'Person') for kind in ('ctxt', 'lex')},
    **{f'Exponence[{name}]': f'Exponence__{name}' for name in ('Gender', 'Number', 'Person')},
    'Tense[denom]': 'Tense__denom', 'Status[Tense]': 'Status__Tense',
}
_EXTERNAL_NAMES = {value: key for key, value in _MISC_NAMES.items()}


class RewriteGraph:
    def __init__(self, sentence):
        self.sentence = sentence
        self.nodes = {0: {}}
        self.edges = {}
        self.order = [0]
        self.rows = {}
        for token in sentence:
            if type(token['id']) is not int:
                continue
            identifier = token['id']
            self.rows[identifier] = deepcopy(dict(token))
            self.nodes[identifier] = {
                key: token[key] for key in ('form', 'lemma', 'upos', 'xpos')
                if token.get(key) not in (None, '_', '')
            }
            self.nodes[identifier]['wordform'] = token['form']
            for column in ('feats', 'misc'):
                if isinstance(token.get(column), dict):
                    self.nodes[identifier].update({_MISC_NAMES.get(k, k): v for k, v in token[column].items()})
            self.order.append(identifier)
            relation = token.get('deprel')
            if isinstance(token.get('head'), int) and relation not in (None, '_'):
                self.add_edge(token['head'], identifier, parse_label(relation))

    def add_edge(self, source, target, attrs):
        identifier = max(self.edges, default=-1) + 1
        self.edges[identifier] = {'src': source, 'dst': target, 'attrs': dict(attrs)}
        return identifier

    def add_node(self, before=None):
        identifier = max(self.nodes, default=0) + 1
        self.nodes[identifier] = {}
        index = self.order.index(before) if before in self.order else len(self.order)
        self.order.insert(index, identifier)
        return identifier

    def to_sentence(self):
        from .conllu import Token, TokenList
        if set(self.nodes) - {0} != set(self.rows) & set(self.nodes):
            raise ValueError('Rule output still contains temporary nodes.')
        tokens = []
        for original in self.sentence:
            identifier = original['id']
            if type(identifier) is not int:
                if isinstance(identifier, tuple) and len(identifier) == 3 and identifier[1] == '.':
                    continue
                tokens.append(Token(deepcopy(dict(original))))
                continue
            if identifier not in self.nodes:
                continue
            row = deepcopy(self.rows[identifier])
            attrs = self.nodes[identifier]
            for key in ('form', 'lemma', 'upos', 'xpos'):
                row[key] = attrs.get(key, '_')
            original_misc = row.get('misc') or {}
            misc_keys = ({_MISC_NAMES.get(k, k) for k in original_misc}
                         if isinstance(original_misc, dict) else set()) | set(_EXTERNAL_NAMES) | {
                'Subject', 'Idiom', 'InIdiom', 'Title', 'InTitle', 'Coconstruct', 'Backchannel'
            }
            row['feats'] = {k: v for k, v in sorted(attrs.items(), key=lambda item: item[0].casefold())
                            if k not in {'form', 'lemma', 'upos', 'xpos', 'wordform', 'textform'} | misc_keys} or None
            row['misc'] = {_EXTERNAL_NAMES.get(k, k): attrs[k]
                           for k in sorted(misc_keys, key=lambda k: _EXTERNAL_NAMES.get(k, k).casefold()) if k in attrs} or None
            incoming = [e for e in self.edges.values() if e['dst'] == identifier]
            if len(incoming) != 1:
                raise ValueError(f"Rule output has {len(incoming)} incoming edges at token {identifier}")
            row['head'] = incoming[0]['src']
            row['deprel'] = format_label(incoming[0]['attrs'])
            row['deps'] = None
            tokens.append(Token(row))
        return TokenList(tokens, deepcopy(self.sentence.metadata))


def parse_label(label):
    surface, _, deep = label.partition('@')
    base, _, subtype = surface.partition(':')
    result = {'1': base}
    if subtype:
        result['2'] = subtype
    if deep:
        result['deep'] = deep
    return result


def format_label(attrs):
    result = attrs.get('1', '')
    if attrs.get('2'):
        result += ':' + attrs['2']
    if attrs.get('deep'):
        result += '@' + attrs['deep']
    return result


def _convert_sentence(sentence, grs_path=None, *, direction='UD_to_SUD', max_steps=10000, trace=None):
    """Apply the pinned official rules in Python to a copy of one sentence.

    Unsupported rule syntax raises an error rather than silently skipping it.
    Bundled rules are the default; custom rules must retain their imports.
    """
    graph = RewriteGraph(sentence)
    if grs_path is None:
        grs_path = Path(__file__).parent / 'resources' / 'sud' / 'grs' / (direction + '.grs')
    program = _load_program(str(Path(grs_path).resolve()))
    def matcher(current, rule):
        bindings = find_match(current, rule)
        if bindings is None:
            return None
        edge_names = {e[0] for e in rule.pattern.edges if e[0]}
        return {'nodes': {k:v for k,v in bindings.items() if k in rule.pattern.nodes},
                'edges': {k:v for k,v in bindings.items() if k in edge_names}}
    execute_rules(graph, program, direction + '_main', max_steps=max_steps, matcher=matcher, trace=trace)
    return graph.to_sentence()


def _ud_sentence_to_sud(sentence, grs_path=None, **kwargs):
    return _convert_sentence(sentence, grs_path, direction='UD_to_SUD', **kwargs)


def _sud_sentence_to_ud(sentence, grs_path=None, **kwargs):
    return _convert_sentence(sentence, grs_path, direction='SUD_to_UD', **kwargs)


def ud_to_sud(sentence, grs_path=None, *, output_path=None, output_dir=None,
              recursive=False, overwrite=False, max_steps=10000, trace=None):
    """Convert AG sentence containers or CoNLL-U files/directories.

    Memory inputs return the corresponding AG type, without mutation.
    File outputs default to NAME.sud.conllu; directory outputs default to
    a sibling NAME_sud folder. Paths return a Path or a list of Paths.
    Directory conversion preserves relative paths and selects .conllu files.
    Existing destinations require overwrite=True. Batch conversion stops on
    error; earlier successfully converted files remain available.
    """
    return _convert_input(sentence, grs_path, output_path=output_path, output_dir=output_dir,
                          recursive=recursive, overwrite=overwrite, max_steps=max_steps, trace=trace)


def sud_to_ud(sentence, grs_path=None, *, output_path=None, output_dir=None,
              recursive=False, overwrite=False, max_steps=10000, trace=None):
    """Apply official SUD-to-UD rules to memory containers or filesystem inputs.

    Defaults are NAME.ud.conllu or a sibling NAME_ud directory. This is a
    rule-based reverse conversion, not lossless recovery of the original UD.
    Enhanced dependencies and empty nodes are not reconstructed.
    """
    return _convert_input(sentence, grs_path, direction='SUD_to_UD', output_path=output_path,
                          output_dir=output_dir, recursive=recursive, overwrite=overwrite,
                          max_steps=max_steps, trace=trace)


def _convert_input(sentence, grs_path=None, *, direction='UD_to_SUD', output_path=None,
                   output_dir=None, recursive=False, overwrite=False, max_steps=10000, trace=None):
    from .conllu import TokenList, SentenceList, SentenceLists
    suffix = 'sud' if direction == 'UD_to_SUD' else 'ud'
    if output_path == '':
        output_path = None
    if output_dir == '':
        output_dir = None
    options = dict(grs_path=grs_path, direction=direction, max_steps=max_steps, trace=trace)
    if isinstance(sentence, (TokenList, SentenceList, SentenceLists)):
        if output_path is not None or output_dir is not None or recursive or overwrite:
            raise ValueError('Output and directory options apply only to path inputs.')
        if isinstance(sentence, TokenList):
            return _convert_sentence(sentence, **options)
        if isinstance(sentence, SentenceList):
            return SentenceList([_convert_sentence(s, **options) for s in sentence],
                                metadata=deepcopy(sentence.metadata))
        return SentenceLists([_convert_input(document, **options) for document in sentence])
    if not isinstance(sentence, (str, os.PathLike)):
        raise TypeError('Expected TokenList, SentenceList, SentenceLists, or a filesystem path.')
    if trace is not None:
        raise ValueError('trace is supported for memory inputs only.')
    source = Path(sentence).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f'Input not found: {source}')
    if source.is_file():
        if output_dir is not None or recursive:
            raise ValueError('Use output_path for a single file.')
        destination = (Path(output_path).expanduser().resolve() if output_path is not None
                       else source.with_name(source.stem + '.' + suffix + '.conllu'))
        pairs = [(source, destination)]
    elif source.is_dir():
        if output_path is not None:
            raise ValueError('Use output_dir for a directory.')
        destination = (Path(output_dir).expanduser().resolve() if output_dir is not None
                       else source.with_name(source.name + '_' + suffix))
        if destination == source or source in destination.parents:
            raise ValueError('Output directory must be outside the input directory.')
        if destination.exists() and not destination.is_dir():
            raise ValueError(f'Output directory is not a directory: {destination}')
        files = sorted(p for p in (source.rglob('*') if recursive else source.iterdir())
                       if p.is_file() and p.suffix.lower() == '.conllu')
        pairs = [(path, destination / path.relative_to(source)) for path in files]
    else:
        raise ValueError(f'Unsupported input path: {source}')
    # Validate all destinations before writing any batch output.
    for path, target in pairs:
        if path == target or (target.exists() and os.path.samefile(path, target)):
            raise ValueError('Input and output must be different files.')
        if target.exists() and not overwrite:
            raise FileExistsError(f'Output already exists: {target}')
        if target.exists() and not target.is_file():
            raise ValueError(f'Output is not a file: {target}')
        for parent in target.parents:
            if parent.exists():
                if not parent.is_dir():
                    raise ValueError(f'Output parent is not a directory: {parent}')
                break
    results = []
    for path, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        results.append(_convert_path_file(path, target, grs_path=grs_path, direction=direction, max_steps=max_steps))
    return results[0] if source.is_file() else results


@lru_cache(maxsize=8)
def _load_program(path):
    return parse_grs(path)

"""Dependency annotation scheme conversion helpers."""

from pathlib import Path
from copy import deepcopy
from io import StringIO
from itertools import chain
import os
import tempfile
from typing import Iterable

from .conllu import SentenceList, SentenceLists, TokenList, parse_incr
from .tree_manipulation import convert_ud_function_dependents, restore_ud_function_dependents
SUD_CORE_FUNCTION_DEPRELS = ("case",)
SUD_EXPERIMENTAL_FUNCTION_DEPRELS = ("case", "aux", "aux:pass", "mark", "cop")


def ud_to_sud(
    data,
    *,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    overwrite: bool = False,
    target_deprels: Iterable[str] | None = None,
    in_place: bool = False,
    max_steps: int = 10000,
    trace: list | None = None,
):
    """Convert UD to SUD using bundled ordinary Python rule functions.

    No GRS file, interpreter, external executable, or model is needed.
    Explicit target_deprels selects legacy local alternations instead.
    Full conversion clears enhanced DEPS and empty nodes, without lossless
    round-trip guarantees. Memory inputs are copied unless in_place=True.
    """
    target_deprels = None if target_deprels is None else tuple(target_deprels)
    if trace is not None and not isinstance(data, (TokenList, SentenceList, SentenceLists)):
        raise ValueError('trace is supported for memory inputs only.')
    return _convert_input(
        data,
        direction="ud_to_sud",
        transform=lambda item: _transform(item, 'UD_to_SUD', target_deprels, in_place, max_steps, trace),
        output_path=output_path,
        output_dir=output_dir,
        recursive=recursive,
        overwrite=overwrite,
    )


def sud_to_ud(
    data,
    *,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    overwrite: bool = False,
    target_deprels: Iterable[str] | None = None,
    in_place: bool = False,
    max_steps: int = 10000,
    trace: list | None = None,
):
    """Convert SUD to UD with Python rules; not lossless original-UD recovery."""
    target_deprels = None if target_deprels is None else tuple(target_deprels)
    if trace is not None and not isinstance(data, (TokenList, SentenceList, SentenceLists)):
        raise ValueError('trace is supported for memory inputs only.')
    return _convert_input(
        data,
        direction="sud_to_ud",
        transform=lambda item: _transform(item, 'SUD_to_UD', target_deprels, in_place, max_steps, trace),
        output_path=output_path,
        output_dir=output_dir,
        recursive=recursive,
        overwrite=overwrite,
    )


def convert_ud_to_sud(
    input_conllu: str | Path,
    output_conllu: str | Path,
    *,
    overwrite: bool = True,
    target_deprels: Iterable[str] | None = None,
    max_steps: int = 10000,
    backend: str = 'python',
):
    """Convert one UD file using Python rules and an atomic output replacement."""
    if backend != 'python':
        raise ValueError("Only backend='python' is supported; GRS is not used.")
    return ud_to_sud(
        input_conllu,
        output_path=output_conllu,
        overwrite=overwrite,
        target_deprels=target_deprels,
        max_steps=max_steps,
    )


def convert_sud_to_ud(
    input_conllu: str | Path,
    output_conllu: str | Path,
    *,
    overwrite: bool = True,
    target_deprels: Iterable[str] | None = None,
    max_steps: int = 10000,
    backend: str = 'python',
):
    """Convert one SUD file using Python rules and an atomic output replacement."""
    if backend != 'python':
        raise ValueError("Only backend='python' is supported; GRS is not used.")
    return sud_to_ud(
        input_conllu,
        output_path=output_conllu,
        overwrite=overwrite,
        target_deprels=target_deprels,
        max_steps=max_steps,
    )


def _convert_input(
    data,
    *,
    direction: str,
    transform,
    output_path: str | Path | None,
    output_dir: str | Path | None,
    recursive: bool,
    overwrite: bool,
):
    if output_path == "":
        output_path = None
    if output_dir == "":
        output_dir = None

    if isinstance(data, (TokenList, SentenceList, SentenceLists)):
        if output_path is not None or output_dir is not None or recursive or overwrite:
            raise ValueError("Output and directory options apply only to path inputs.")
        return transform(data)

    if not isinstance(data, (str, Path)):
        raise TypeError("data must be a TokenList, SentenceList, SentenceLists, file path, or directory path")

    source = Path(data).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")

    suffix = "sud" if direction == "ud_to_sud" else "ud"
    if source.is_file():
        if output_dir is not None or recursive:
            raise ValueError("Use output_path for a single file.")
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else source.with_name(source.stem + "." + suffix + ".conllu")
        )
        pairs = [(source, destination)]
    elif source.is_dir():
        if output_path is not None:
            raise ValueError("Use output_dir for a directory.")
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else source.with_name(source.name + "_" + suffix)
        )
        if destination == source or source in destination.parents:
            raise ValueError("Output directory must be outside the input directory.")
        files = sorted(
            path for path in (source.rglob("*") if recursive else source.iterdir())
            if path.is_file() and path.suffix.lower() == ".conllu"
        )
        pairs = [(path, destination / path.relative_to(source)) for path in files]
    else:
        raise ValueError(f"Unsupported input path: {source}")

    for path, target in pairs:
        if path == target or (target.exists() and path.samefile(target)):
            raise ValueError("Input and output must be different files.")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {target}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Output is not a file: {target}")

    results = []
    for path, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        _convert_file(path, target, transform)
        results.append(target)
    return results[0] if source.is_file() else results


def _transform(data, direction, targets, in_place, max_steps, trace):
    if targets is not None:
        converter = convert_ud_function_dependents if direction == 'UD_to_SUD' else restore_ud_function_dependents
        return converter(data, target_deprels=targets, in_place=in_place)
    if isinstance(data, TokenList):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer.')
        result = _convert_sentence(data, direction=direction, max_steps=max_steps, trace=trace)
        if in_place:
            from .tree_manipulation import _replace_tokenlist_contents
            return _replace_tokenlist_contents(data, result)
        return result
    if isinstance(data, SentenceList):
        converted = [_transform(s, direction, None, False, max_steps, trace) for s in data]
        result = SentenceList(converted, deepcopy(data.metadata))
    else:
        result = SentenceLists([_transform(s, direction, None, False, max_steps, trace) for s in data])
    if in_place:
        data[:] = result
        return data
    return result


def _convert_file(source, destination, transform):
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / 'converted.conllu'
        with source.open(encoding='utf-8') as reader, temporary.open('w', encoding='utf-8', newline='\n') as writer:
            block = []
            for line in chain(reader, ('\n',)):
                if line.strip():
                    block.append(line.rstrip('\r\n'))
                    continue
                if not block:
                    continue
                sentence = next(parse_incr(StringIO('\n'.join(block) + '\n\n')))
                try:
                    serialized = transform(sentence).serialize()
                except Exception as exc:
                    raise RuntimeError(f"Conversion failed at {sentence.metadata.get('sent_id', '<missing>')}: {exc}") from exc
                comments = [line for line in block if line.startswith('#')]
                rows = [line for line in serialized.splitlines() if line and not line.startswith('#')]
                writer.write('\n'.join(comments + rows) + '\n\n')
                block = []
        os.replace(temporary, destination)
    if isinstance(data, TokenList):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError('max_steps must be a positive integer.')
        result = _convert_sentence(data, direction=direction, max_steps=max_steps, trace=trace)
        if in_place:
            from .tree_manipulation import _replace_tokenlist_contents
            return _replace_tokenlist_contents(data, result)
        return result
    if isinstance(data, SentenceList):
        converted = [_transform(s, direction, None, False, max_steps, trace) for s in data]
        result = SentenceList(converted, deepcopy(data.metadata))
    else:
        result = SentenceLists([_transform(s, direction, None, False, max_steps, trace) for s in data])
    if in_place:
        data[:] = result
        return data
    return result


def _convert_file(source, destination, transform):
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / 'converted.conllu'
        with source.open(encoding='utf-8') as reader, temporary.open('w', encoding='utf-8', newline='\n') as writer:
            block = []
            for line in chain(reader, ('\n',)):
                if line.strip():
                    block.append(line.rstrip('\r\n'))
                    continue
                if not block:
                    continue
                sentence = next(parse_incr(StringIO('\n'.join(block) + '\n\n')))
                try:
                    serialized = transform(sentence).serialize()
                except Exception as exc:
                    raise RuntimeError(f"Conversion failed at {sentence.metadata.get('sent_id', '<missing>')}: {exc}") from exc
                comments = [line for line in block if line.startswith('#')]
                rows = [line for line in serialized.splitlines() if line and not line.startswith('#')]
                writer.write('\n'.join(comments + rows) + '\n\n')
                block = []
        os.replace(temporary, destination)
