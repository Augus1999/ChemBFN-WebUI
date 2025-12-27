# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
Testing prompt parsing functionalities.
"""
import pytest
from chembfn_webui.lib.utilities import (
    parse_prompt,
    parse_exclude_token,
    parse_sar_control,
)


@pytest.mark.parametrize(
    "input_value,vocab_keys,expected",
    [
        (None, ["<start>", "<end>", "C"], []),
        ("", ["C", "<start>", "<pad>"], []),
        ("C,c,1", ["C", "N", "S", "c", "n", "1"], ["N", "S", "n"]),
        ("[C], [=Branch1]", ["[C]", "[=C]", "[=Branch1]", "[N]"], ["[=C]", "[N]"]),
    ],
)
def test_parse_exclude_token(input_value, vocab_keys, expected):
    assert parse_exclude_token(input_value, vocab_keys) == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (None, [False]),
        ("", [False]),
        ("F,T", [False, True]),
        ("t,f, T", [True, False, True]),
        ("abcdef", [False]),
    ],
)
def test_parse_sar_flag(input_value, expected):
    assert parse_sar_control(input_value) == expected


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (None, {"lora": [], "objective": [], "lora_scaling": []}),
        ("", {"lora": [], "objective": [], "lora_scaling": []}),
        ("abcdsfs", {"lora": [], "objective": [], "lora_scaling": []}),
        ("0.5", {"lora": [], "objective": [[0.5]], "lora_scaling": []}),
        ("[0.5, 0.3]", {"lora": [], "objective": [[0.5, 0.3]], "lora_scaling": []}),
        ("<lora:0.5>", {"lora": ["lora"], "objective": [], "lora_scaling": [0.5]}),
        ("<lora>", {"lora": ["lora"], "objective": [], "lora_scaling": [1.0]}),
        (
            "<lora:0.12w>:[0.3,0.5]",
            {"lora": ["lora"], "objective": [[0.3, 0.5]], "lora_scaling": [1.0]},
        ),
        (
            "<lora1:0.1>:[0.5,1.0];<lora2:0.9>:[10]",
            {
                "lora": ["lora1", "lora2"],
                "objective": [[0.5, 1.0], [10]],
                "lora_scaling": [0.1, 0.9],
            },
        ),
        (
            "<lora1:0.1>:[0.5,1.0]; <lora2:0.9>:[10]",
            {
                "lora": ["lora1", "lora2"],
                "objective": [[0.5, 1.0], [10]],
                "lora_scaling": [0.1, 0.9],
            },
        ),
        (
            "<lora1:0.1>:[0.5, 1.0];\n<lora2:0.9>:[10];\n<lora3>:[-0.1, qs]",
            {
                "lora": ["lora1", "lora2", "lora3"],
                "objective": [[0.5, 1.0], [10]],
                "lora_scaling": [0.1, 0.9, 1.0],
            },
        ),
    ],
)
def test_parse_prompt(input_value, expected):
    assert parse_prompt(input_value) == expected
