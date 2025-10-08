# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
Define application behaviours.
"""
import sys
import argparse
from pathlib import Path
from typing import Tuple, List

sys.path.append(str(Path(__file__).parent.parent))
from rdkit.Chem import Draw, MolFromSmiles
from mol2chemfigPy3 import mol2chemfig
import gradio as gr
import torch
from lib.utilities import find_model, find_vocab

vocabs = find_vocab()
models = find_model()


def refresh() -> Tuple:
    global vocabs, models
    vocabs = find_vocab()
    models = find_model()
    a = list(vocabs.keys())
    b = [i[0] for i in models["base"]]
    c = [[i[0], i[2]] for i in models["standalone"]]
    d = [[i[0], i[2]] for i in models["lora"]]
    return a, b, c, d


def mols2chemfig(mols: str) -> str:
    mols = mols.split("\n")
    return "\n".join([mol2chemfig(i, "r", inline=True) for i in mols])


def dummy(a, b, c, d, e, f, g):
    return [Draw.MolToImage(MolFromSmiles(""))], ["111"] * 50, ["\chemfig{}"] * 50


with gr.Blocks(title="ChemBFN WebUI") as app:
    gr.Markdown("### WebUI to generate and visualise molecules for ChemBFN method.")
    gr.Markdown("Author: Nianze TAO (Omozawa SUENO)")
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            model_name = gr.Dropdown(
                [i[0] for i in models["base"]] + [i[0] for i in models["standalone"]],
                label="model",
            )
            batch_size = gr.Slider(1, 512, 1, step=1, label="batch size")
            sequence_size = gr.Slider(5, 4096, None, step=1, label="sequence length")
            method = gr.Dropdown(["BFN", "ODE"], label="method")
            temperature = gr.Slider(0.0, 2.5, 0.5, step=0.001, label="temperature")
            btn = gr.Button("RUN", variant="primary")
            img = gr.Gallery(label="molecule")
        with gr.Column(scale=2):
            with gr.Tab(label="prompt editor"):
                prompt = gr.TextArea(label="prompt")
                scaffold = gr.Textbox(label="scaffold")
            with gr.Tab(label="result viewer"):
                result = gr.Dataframe(
                    headers=["molecule"],
                    col_count=(1, "fixed"),
                    label="result",
                    interactive=False,
                    show_row_numbers=True,
                )
                chemfig = gr.Dataframe(
                    headers=["code"],
                    col_count=(1, "fixed"),
                    label="LATEX ChemFig",
                    interactive=False,
                    show_row_numbers=True,
                )
            with gr.Tab(label="model explorer"):
                btn_refresh = gr.Button("refresh", variant="secondary")
                vocab_table = gr.Dataframe(
                    list(vocabs.keys()),
                    headers=["name"],
                    col_count=(1, "fixed"),
                    label="vocabulary files",
                    interactive=False,
                    show_row_numbers=True,
                )
                base_table = gr.Dataframe(
                    [i[0] for i in models["base"]],
                    headers=["name"],
                    col_count=(1, "fixed"),
                    label="base models",
                    interactive=False,
                    show_row_numbers=True,
                )
                standalone_table = gr.Dataframe(
                    [[i[0], i[2]] for i in models["standalone"]],
                    headers=["name", "objective"],
                    col_count=(2, "fixed"),
                    label="standalone models",
                    interactive=False,
                    show_row_numbers=True,
                )
                lora_tabel = gr.Dataframe(
                    [[i[0], i[2]] for i in models["lora"]],
                    headers=["name", "objective"],
                    col_count=(2, "fixed"),
                    label="LoRA models",
                    interactive=False,
                    show_row_numbers=True,
                )
    btn.click(
        fn=dummy,
        inputs=[
            model_name,
            batch_size,
            sequence_size,
            method,
            temperature,
            prompt,
            scaffold,
        ],
        outputs=[img, result, chemfig],
    )
    btn_refresh.click(
        fn=refresh,
        inputs=None,
        outputs=[vocab_table, base_table, standalone_table, lora_tabel],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public", default=False, help="open to public", action="store_true"
    )
    args = parser.parse_args()
    app.launch(share=args.public)
