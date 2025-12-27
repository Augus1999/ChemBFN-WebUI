# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
Utilities.
"""
import os
import ast
import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable, Any

_model_path = Path(__file__).parent.parent / "model"
if "CHEMBFN_WEBUI_MODEL_DIR" in os.environ:
    _model_path = Path(os.environ["CHEMBFN_WEBUI_MODEL_DIR"])

_ALLOWED_STRING_METHODS = {"strip", "replace", "split"}
_ALLOWED_NODES = (
    ast.arguments,
    ast.arg,
    ast.Expression,
    ast.Attribute,
    ast.Subscript,
    ast.Constant,
    ast.UnaryOp,
    ast.Lambda,
    ast.Load,
    ast.Name,
    ast.Call,
    ast.USub,
)


class _SafeLambdaValidator(ast.NodeVisitor):
    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        super().visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        if len(node.args.args) != 1:
            raise ValueError("Only one argument is accepted")
        if node.args.args[0].arg != "x":
            raise ValueError("Lambda argument must be named 'x'")
        self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id != "x":
            raise ValueError(f"Only variable 'x' is allowed, not '{node.id}'")

    def visit_arguments(self, node: ast.arguments) -> None:
        if len(node.args) > 1:
            raise ValueError("Only one argument is allowed")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        # Only allow x.split(...)[idx]
        if not isinstance(node.value, ast.Call):
            raise ValueError("Indexing should only be used after `split` method")
        if not isinstance(node.value.func, ast.Attribute):
            raise ValueError("Indexing should only be used after `split` method")
        if node.value.func.attr != "split":
            raise ValueError("Indexing should only be used after `split` method")
        if not isinstance(node.slice, (ast.Constant, ast.UnaryOp)):
            raise ValueError("Only number index is accepted")
        self.visit(node.value)
        idx = node.slice  # should be positive or negative int
        if isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.USub):
            if not isinstance(idx.operand, ast.Constant):
                raise ValueError("Invalid index")
        elif not isinstance(idx, ast.Constant):
            raise ValueError("Index must be an integer literal")
        self.visit(idx)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        # Only allow x.<method>
        if not isinstance(node.value, ast.Name):
            raise ValueError("No nested method calling is allowed")
        if node.value.id != "x":
            raise ValueError("Please only use 'x' as argument")
        if node.attr not in _ALLOWED_STRING_METHODS:
            raise ValueError(f"Method '{node.attr}' not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        # only allow x.<method>(...)
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Only method calls on string objects are allowed")
        self.visit(node.func)
        for arg in node.args:
            if not isinstance(arg, ast.Constant):
                raise ValueError("Only literal arguments allowed")
            self.visit(arg)
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed")


def sys_info() -> str:
    """
    Get system information.

    :return: system info in html format
    :rtype: str
    """
    import sys
    import torch
    import gradio as gr
    import bayesianflow_for_chem as bfn
    from .version import __version__

    _python_version = ".".join([str(i) for i in sys.version_info[:3]])
    return f"""
            version: <a href="https://github.com/Augus1999/ChemBFN-WebUI">{__version__}</a>
            &#x2000;•&#x2000;
            bayesianflow-for-chem: <a href="https://github.com/Augus1999/bayesian-flow-network-for-chemistry">{bfn.__version__}</a>
            &#x2000;•&#x2000;
            python: {_python_version}
            &#x2000;•&#x2000;
            torch: {getattr(torch, '__long_version__', torch.__version__)}
            &#x2000;•&#x2000;
            gradio: {gr.__version__}
            """


def find_vocab() -> Dict[str, str]:
    """
    Find customised vocabulary files.

    :return: {file_name: file_path}
    :rtype: dict
    """
    vocab_fns = glob(str(_model_path / "vocab/*.txt"))
    return {
        os.path.basename(i).replace(".txt", ""): i
        for i in vocab_fns
        if "place_vocabulary_file_here.txt" not in i
    }


def find_model() -> Dict[str, List[List[Union[str, int, List[str], Path]]]]:
    """
    Find model files.

    :return: ```
            {
              "base": [[name1, path1], [name2, path2], ...],
              "standalone": [[name1, parent_path1, label1, pad_len1], ...],
              "lora": [[name1, parent_path1, label1, pad_len1], ...]
            }```
    :rtype: dict
    """
    models = {}
    # find base models
    base_fns = glob(str(_model_path / "base_model/*.pt"))
    models["base"] = [[os.path.basename(i), i] for i in base_fns]
    # find standalone models
    standalone_models = []
    standalone_fns = glob(str(_model_path / "standalone_model/*/model.pt"))
    for standalone_fn in standalone_fns:
        config_fn = Path(standalone_fn).parent / "config.json"
        if not os.path.exists(config_fn):
            continue
        else:
            with open(config_fn, "r", encoding="utf-8") as f:
                config = json.load(f)
            name = config["name"]
            label = config["label"]
            lmax = config["padding_length"]
            standalone_models.append([name, Path(standalone_fn).parent, label, lmax])
    models["standalone"] = standalone_models
    # find lora models
    lora_models = []
    lora_fns = glob(str(_model_path / "lora/*/lora.pt"))
    for lora_fn in lora_fns:
        config_fn = Path(lora_fn).parent / "config.json"
        if not os.path.exists(config_fn):
            continue
        else:
            with open(config_fn, "r", encoding="utf-8") as f:
                config = json.load(f)
            name = config["name"]
            label = config["label"]
            lmax = config["padding_length"]
            lora_models.append([name, Path(lora_fn).parent, label, lmax])
    models["lora"] = lora_models
    return models


def _get_lora_info(prompt: str) -> Tuple[str, List[float], float]:
    """
    Parse sub-prompt string containing LoRA info.

    :param prompt: LoRA sub-pompt: \n
                   case I. `"<name:A>"` \n
                   case II. `"<name>"` \n
                   case III. `"<name:A>:[a,b,...]"` \n
                   case IV. `"<name>:[a,b,c,...]"`
    :type prompt: str
    :return: LoRA name \n
             objective values \n
             LoRA scaling
    :rtype: tuple
    """
    s = prompt.split(">")
    s1 = s[0].replace("<", "")
    lora_info = s1.split(":")
    lora_name = lora_info[0]
    if len(lora_info) == 1:
        lora_scaling = 1.0
    else:
        try:
            lora_scaling = float(lora_info[1])
        except ValueError as error:
            print(f"{error}. Reset `lora_scaling` to 1.0.")
            lora_scaling = 1.0
    if len(s) == 1:
        obj = []
    elif ":" not in s[1]:
        obj = []
    else:
        s2 = s[1].replace(":", "").replace("[", "").replace("]", "").split(",")
        try:
            obj = [float(i) for i in s2]
        except ValueError as error:
            print(f"{error}. Reset `obj` to empty.")
            obj = []
    return lora_name, obj, lora_scaling


def parse_prompt(
    prompt: Optional[str],
) -> Dict[str, Union[List[str], List[float], List[List[float]]]]:
    """
    Parse propmt.

    :param prompt: prompt string: \n
                   case I. empty string `""` --> `{"lora": [], "objective": [], "lora_scaling": []}`\n
                   case II. one condition `"[a,b,c,...]"` --> `{"lora": [], "objective": [[a, b, c, ...]], "lora_scaling": []}`\n
                   case III. one LoRA `"<name:A>"` --> `{"lora": [name], "objective": [], "lora_scaling": [A]}`\n
                   case IV. one LoRA `"<name>"` --> `{"lora": [name], "objective": [], "lora_scaling": [1]}`\n
                   case V. one LoRA with condition `"<name:A>:[a,b,...]"` --> `{"lora": [name], "objective": [[a, b, ...]], "lora_scaling": [A]}`\n
                   case VI. one LoRA with condition `"<name>:[a,b,...]"` --> `{"lora": [name], "objective": [[a, b, ...]], "lora_scaling": [1]}`\n
                   case VII. several LoRAs with conditions `"<name1:A1>:[a1,b1,...];<name2>:[a2,b2,c2,...]"` --> `{"lora": [name1, name2], "objective": [[a1, b1, ...], [a2, b2, c2, ...]], "lora_scaling": [A1, 1]}`\n
                   case VIII. other cases --> `{"lora": [], "objective": [], "lora_scaling": []}`\n
    :type prompt: str | None
    :return: ```
            {
              "lora": [name1, name2, ...],
              "objective": [obj1, obj2, ...],
              "lora_scaling": [s1, s2, ...]
            }```
    :rtype: dict
    """
    if prompt is None:
        prompt = ""
    prompt_group = prompt.strip().replace("\n", "").split(";")
    prompt_group = [i.strip() for i in prompt_group if i.strip()]
    info = {"lora": [], "objective": [], "lora_scaling": []}
    if not prompt_group:
        pass
    if len(prompt_group) == 1:
        if not ("<" in prompt_group[0] and ">" in prompt_group[0]):
            try:
                obj = [
                    float(i)
                    for i in prompt_group[0]
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                ]
                info["objective"].append(obj)
            except ValueError as error:
                print(f"{error}. Reset `obj` to empty.")
        else:
            lora_name, obj, lora_scaling = _get_lora_info(prompt_group[0])
            info["lora"].append(lora_name)
            if obj:
                info["objective"].append(obj)
            info["lora_scaling"].append(lora_scaling)
    else:
        for _prompt in prompt_group:
            if not ("<" in _prompt and ">" in _prompt):
                continue
            lora_name, obj, lora_scaling = _get_lora_info(_prompt)
            info["lora"].append(lora_name)
            if obj:
                info["objective"].append(obj)
            info["lora_scaling"].append(lora_scaling)
    return info


def parse_exclude_token(tokens: Optional[str], vocab_keys: List[str]) -> List[str]:
    """
    Parse exclude token string.

    :param tokens: unwanted token string in the format `"token1,token2,..."`
    :param vocab_keys: vocabulary elements
    :type tokens: str | None
    :type vocab_keys: list
    :return: a list of allowed vocabulary
    :rtype: list
    """
    if tokens is None:
        tokens = ""
    tokens = tokens.strip().replace("\n", "").split(",")
    tokens = [i.strip() for i in tokens if i.strip()]
    if not tokens:
        return tokens
    tokens = [i for i in vocab_keys if i not in tokens]
    return tokens


def parse_sar_control(sar_control: Optional[str]) -> List[bool]:
    """
    Parse semi-autoregression control string.

    :param sar_control: semi-autoregression control string: \n
                        case I. `""` --> `[False]` \n
                        case II. `"F"` --> `[False]` \n
                        case III. `"T"` --> `[True]` \n
                        case IV. `F,T,...` --> `[False, True, ...]` \n
                        case V. other cases --> `[False, False, ...]` \n
    :type sar_control: str | None
    :return: a list of SAR flag
    :rtype: list
    """
    if sar_control is None:
        sar_control = ""
    sar_flag = sar_control.strip().replace("\n", "").split(",")
    sar_flag = [i.strip() for i in sar_flag if i.strip()]
    if not sar_flag:
        return [False]
    sar_flag = [i.lower() == "t" for i in sar_flag]
    return sar_flag


def build_result_prep_fn(fn_string: Optional[str]) -> Callable[[str], str]:
    """
    Build result preprocessing function.

    :param fn_string: string form result preprocessing function
    :type fn_string: str | None
    :return: Description
    :rtype: callable
    """
    if not fn_string:
        return lambda x: x
    try:
        tree = ast.parse(fn_string, mode="eval")
        _SafeLambdaValidator().visit(tree)
        code = compile(tree, filename="<safe_lambda>", mode="eval")
        fn = eval(code, {"__builtins__": {}}, {})
        if not callable(fn):
            print(
                "Warning: Expression did not produce a function. Returned identity as result preprocessing function."
            )
            return lambda x: x
        return fn
    except Exception as e:
        print(
            f"Invalid or unsafe expression: {e}. Returned identity as result preprocessing function."
        )
        return lambda x: x


if __name__ == "__main__":
    ...
