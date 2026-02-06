# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
Folder structures.
"""
from pathlib import Path


def _mk_readme_file(fn: Path, content: str) -> None:
    with open(fn, "w", encoding="utf-8") as f:
        f.write(content)


def create_model_dir(user_path: Path) -> None:
    """
    Create an empty directory to hold the model files.

    :param user_path: user provided path
    :type user_path: pathlib.Path
    :return:
    :rtype: None
    """
    if (md := user_path / "model").exists():
        print(f"{str(md)} already exists.")
        return
    Path.mkdir(md)
    Path.mkdir(bmd := md / "base_model")
    Path.mkdir(ld := md / "lora")
    Path.mkdir(smd := md / "standalone_model")
    Path.mkdir(vd := md / "vocab")
    _mk_readme_file(
        bmd / "place_base_model_here.txt", "Place thy base model weight files here."
    )
    _mk_readme_file(
        ld / "place_lora_folder_here.txt",
        "Place thy LoRA weight and configureation files under subfolders here.",
    )
    _mk_readme_file(
        smd / "place_standalone_model_folder_here.txt",
        "Place thy standalone model weight and configuration files under subfloders here.",
    )
    _mk_readme_file(
        vd / "place_vocabulary_file_here.txt", "Place vocabulary files here."
    )


if __name__ == "__main__":
    ...
