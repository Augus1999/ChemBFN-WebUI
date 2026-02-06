# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
Program created model directory should follow the defined structure.
"""
from shutil import rmtree
from pathlib import Path
from chembfn_webui.lib.structs import create_model_dir


cwd = Path(__file__).parent


def test():
    create_model_dir(cwd)
    md = cwd / "model"
    assert (md / "base_model/place_base_model_here.txt").exists()
    assert (md / "lora/place_lora_folder_here.txt").exists()
    assert (md / "standalone_model/place_standalone_model_folder_here.txt").exists()
    assert (md / "vocab/place_vocabulary_file_here.txt").exists()
    rmtree(md)
