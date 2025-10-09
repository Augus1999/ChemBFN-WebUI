# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
Utilities.
"""
import os
import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union

_model_path = Path(__file__).parent.parent / "model"
if "CHEMBFN_WEBUI_MODEL_DIR" in os.environ:
    _model_path = Path(os.environ["CHEMBFN_WEBUI_MODEL_DIR"])


def find_vocab() -> Dict[str, str]:
    vocab_fns = glob(str(_model_path / "vocab/*.txt"))
    return {
        os.path.basename(i).replace(".txt", ""): i
        for i in vocab_fns
        if "place_vocabulary_file_here.txt" not in i
    }


def find_model() -> Dict[str, List[List[Union[str, int, List[str], Path]]]]:
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


def _get_lora_info(prompt: str) -> Tuple[str, List[float], List[float]]:
    s = prompt.split(">")
    s1 = s[0].replace("<", "")
    lora_info = s1.split(":")
    lora_name = lora_info[0]
    if len(lora_info) == 1:
        lora_scaling = 1.0
    else:
        lora_scaling = float(lora_info[1])
    if len(s) == 1:
        obj = []
    elif ":" not in s[1]:
        obj = []
    else:
        s2 = s[1].replace(":", "").replace("[", "").replace("]", "").split(",")
        obj = [float(i) for i in s2]
    return lora_name, obj, lora_scaling


def parse_prompt(
    prompt: str,
) -> Dict[str, Union[List[str], List[float], List[List[float]]]]:
    prompt_group = prompt.strip().replace("\n", "").split(";")
    prompt_group = [i for i in prompt_group if i]
    info = {"lora": [], "objective": [], "lora_scaling": []}
    if not prompt_group:
        pass
    if len(prompt_group) == 1:
        if not ("<" in prompt_group[0] and ">" in prompt_group[0]):
            obj = [
                float(i)
                for i in prompt_group[0].replace("[", "").replace("]", "").split(",")
            ]
            info["objective"].append(obj)
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


def parse_exclude_token(tokens: str, vocab_keys: List[str]) -> List[str]:
    tokens = tokens.strip().replace("\n", "").split(",")
    tokens = [i for i in tokens if i]
    if not tokens:
        return tokens
    tokens = [i for i in vocab_keys if i not in tokens]
    return tokens

def parse_sar_control(sar_control: str) -> List[bool]:
    sar_flag = sar_control.strip().replace("\n", "").split(",")
    sar_flag = [i for i in sar_flag if i]
    if not sar_flag:
        return [False]
    sar_flag = [i.lower() == "t" for i in sar_flag]
    return sar_flag


if __name__ == "__main__":
    ...
