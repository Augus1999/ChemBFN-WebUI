# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
Utilities.
"""
import os
import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Union

_model_path = Path(__file__).parent.parent / "model"


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


def parse_prompt(
    prompt: str,
) -> Dict[str, Union[List[str], List[float], List[List[float]]]]:
    prompt_group = prompt.strip().replace("\n", "").split(";")
    prompt_group = [i for i in prompt_group if i]
    info = {"lora": [], "objective": [], "lora_scaling": []}
    print(not (info["lora"] and info["objective"]))
    print(prompt_group)
    if not prompt_group:
        return info
    if len(prompt_group) == 1:
        if not ("<" in prompt_group[0] and ">" in prompt_group[0]):
            obj = [
                float(i)
                for i in prompt_group[0].replace("[", "").replace("]", "").split(",")
            ]
            info["objective"].append(obj)
            return info
        else:
            ...
    else:
        ...


if __name__ == "__main__":
    parse_prompt("")
    print(parse_prompt("[0,0,0]"))
    parse_prompt(";<n:1:4>;<b:1:5>")
    ...
