# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
`build_result_prep_fn` should only execute safe code.
"""
import ast
import pytest
from chembfn_webui.lib.utilities import _SafeLambdaValidator, build_result_prep_fn


@pytest.mark.parametrize(
    "expr,input_value,expected",
    [
        (None, "abc", "abc"),
        ("lambda x: x.strip()", "  a  ", "a"),
        ("lambda x: x.replace('a', 'b')", "aax", "bbx"),
        ("lambda x: x.split(',')[0]", "a,b,c", "a"),
        ("lambda x: x.split(',')[-1]", "a,b,c", "c"),
        ("lambda x: x.split('>')[1]", "a>b>c", "b"),
        ("lambda x: x.split(',')[True]", "a,b", "b"),
    ],
)
def test_valid_expressions(expr, input_value, expected):
    fn = build_result_prep_fn(expr)
    assert fn(input_value) == expected


@pytest.mark.parametrize(
    "expr,input_value,expected",
    [
        ("x.lower()", "a", "a"),  # not a lambda
        ("lambda x, y: x", "a", "a"),  # too many args
        ("lambda y: y.lower()", "a", "a"),  # wrong arg name
        ("lambda x: x + 'a'", "bc", "bc"),  # operator
        ("lambda x: (x)", "a", "a"),  # parentheses
        ("lambda x: [x]", "a", "a"),  # list literal
        ("lambda x: {'a': x}", "b", "b"),  # dict literal
        ("lambda x: x if x == 'b' else 'c'", "a", "a"),  # conditional
    ],
)
def test_invalid_expressions(expr, input_value, expected):
    fn = build_result_prep_fn(expr)
    assert fn(input_value) == expected


@pytest.mark.parametrize(
    "expr,input_value,expected",
    [
        ("lambda x: x.__class__", "a", "a"),
        ("lambda x: x.__dict__", "a", "a"),
        ("lambda x: x.lower.__call__", "a", "a"),
        ("lambda x: x.split.__globals__", "a", "a"),
        ("lambda x: x.split('a').__class__", "a", "a"),
    ],
)
def test_attribute_abuse(expr, input_value, expected):
    fn = build_result_prep_fn(expr)
    assert fn(input_value) == expected


@pytest.mark.parametrize(
    "expr,input_value,expected",
    [
        ("lambda x: y", "a", "a"),
        ("lambda x: x.split(y)", "a", "a"),
        ("lambda x: x.replace(a, b)", "a", "a"),
        ("lambda x: x.split(',')[i]", "a", "a"),
    ],
)
def test_invalid_name_access(expr, input_value, expected):
    fn = build_result_prep_fn(expr)
    assert fn(input_value) == expected


@pytest.mark.parametrize(
    "expr,input_value,expected",
    [
        ("lambda x: x.lower()", "ABC", "ABC"),
        ("lambda x: x.upper()", "abc", "abc"),
        ("lambda x: x.encode()", "a", "a"),
        ("lambda x: x.format()", r"{}", r"{}"),
        ("lambda x: x.join(['a'])", "a", "a"),  # wrong receiver
        ("lambda x: x.split(sep=',')", "a", "a"),  # keyword args
    ],
)
def test_invalid_methods(expr, input_value, expected):
    fn = build_result_prep_fn(expr)
    assert fn(input_value) == expected


@pytest.mark.parametrize(
    "expr,input_value,expected",
    [
        ("lambda x: x.split(',')[0].lower()", "A,a", "A,a"),
        ("lambda x: x.replace('a', 'b').strip()", "a", "a"),
        ("lambda x: x.split(',').pop()", "a", "a"),
    ],
)
def test_method_chaining(expr, input_value, expected):
    fn = build_result_prep_fn(expr)
    assert fn(input_value) == expected


@pytest.mark.parametrize(
    "expr,input_value,expected",
    [
        ("lambda x: x.strip()[0]", "ab", "ab"),  # indexing non-split
        ("lambda x: x.split(',')[1:]", "a,b", "a,b"),  # slicing
        ("lambda x: x.split(',')[1+1]", "a,b", "a,b"),  # expression index
    ],
)
def test_invalid_subscripts(expr, input_value, expected):
    fn = build_result_prep_fn(expr)
    assert fn(input_value) == expected


@pytest.mark.parametrize(
    "expr,input_value,expected",
    [
        ("lambda x: __import__('os').system('id')", "a", "a"),
        ("lambda x: eval('x')", "a", "a"),
        ("lambda x: open('file')", "a", "a"),
        ("lambda x: globals()", "a", "a"),
        ("lambda x: locals()", "a", "a"),
    ],
)
def test_builtin_escape(expr, input_value, expected):
    fn = build_result_prep_fn(expr)
    assert fn(input_value) == expected


def test_unknown_ast_node_fails_closed():
    class DummyNode(ast.AST):
        pass

    validator = _SafeLambdaValidator()
    with pytest.raises(ValueError):
        validator.visit(DummyNode())
