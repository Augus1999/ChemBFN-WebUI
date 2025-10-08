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
from rdkit.Chem import Draw, MolFromSmiles, MolFromFASTA
from mol2chemfigPy3 import mol2chemfig
import gradio as gr
import torch
from lib.utilities import find_model, find_vocab

vocabs = find_vocab()
models = find_model()


def refresh(
    model_selected: str, vocab_selected: str, tokeniser_selected: str
) -> Tuple[
    List[str], List[str], List[List[str]], List[List[str]], gr.Dropdown, gr.Dropdown
]:
    global vocabs, models
    vocabs = find_vocab()
    models = find_model()
    a = list(vocabs.keys())
    b = [i[0] for i in models["base"]]
    c = [[i[0], i[2]] for i in models["standalone"]]
    d = [[i[0], i[2]] for i in models["lora"]]
    e = gr.Dropdown(
        [i[0] for i in models["base"]] + [i[0] for i in models["standalone"]],
        value=model_selected,
        label="model",
    )
    f = gr.Dropdown(
        list(vocabs.keys()),
        value=vocab_selected,
        label="vocabulary",
        visible=tokeniser_selected == "SELFIES",
    )
    return a, b, c, d, e, f


def dummy(a, b, c, d, e, f, g, h, i, j):
    smi = "c1ccc(OCCC#N)cc1O"
    smis = [smi] * e
    img = [Draw.MolToImage(MolFromSmiles(i)) for i in smis]
    # chemfig_ = [f"```latex\n{mol2chemfig(i, "-r", inline=True)}\n```" for i in smis]
    chemfig_ = "\n\n".join([mol2chemfig(i, "-r", inline=True) for i in smis])
    return img, smis, chemfig_, "message!"


with gr.Blocks(title="ChemBFN WebUI") as app:
    gr.Markdown("### WebUI to generate and visualise molecules for ChemBFN method.")
    # gr.Markdown("Author: Nianze TAO (Omozawa SUENO)")
    with gr.Row():
        with gr.Column(scale=1):
            btn = gr.Button("RUN", variant="primary")
            model_name = gr.Dropdown(
                [i[0] for i in models["base"]] + [i[0] for i in models["standalone"]],
                label="model",
            )
            token_name = gr.Dropdown(
                ["SMILES & SAFE", "SELFIES", "FASTA"], label="tokeniser"
            )
            vocab_fn = gr.Dropdown(
                list(vocabs.keys()),
                label="vocabulary",
                visible=token_name.value == "SELFIES",
            )
            step = gr.Slider(1, 5000, 100, step=1, label="step")
            batch_size = gr.Slider(1, 512, 1, step=1, label="batch size")
            sequence_size = gr.Slider(5, 4096, None, step=1, label="sequence length")
            method = gr.Dropdown(["BFN", "ODE"], label="method")
            temperature = gr.Slider(
                0.0,
                2.5,
                0.5,
                step=0.001,
                label="temperature",
                visible=method.value == "ODE",
            )
        with gr.Column(scale=2):
            with gr.Tab(label="prompt editor"):
                prompt = gr.TextArea(label="prompt", lines=12)
                scaffold = gr.Textbox(label="scaffold")
                gr.Markdown("")
                message = gr.TextArea(label="message")
            with gr.Tab(label="result viewer"):
                with gr.Tab(label="result"):
                    result = gr.Dataframe(
                        headers=["molecule"],
                        col_count=(1, "fixed"),
                        label="",
                        show_fullscreen_button=True,
                        show_row_numbers=True,
                        show_copy_button=True,
                    )
                with gr.Tab(label="LATEX Chemfig"):
                    chemfig = gr.Code(
                        label="", language="latex", show_line_numbers=True
                    )
            with gr.Tab(label="gallery"):
                img = gr.Gallery(label="molecule", columns=4, height=512)
            with gr.Tab(label="model explorer"):
                btn_refresh = gr.Button("refresh", variant="secondary")
                with gr.Tab(label="customised vocabulary"):
                    vocab_table = gr.Dataframe(
                        list(vocabs.keys()),
                        headers=["name"],
                        col_count=(1, "fixed"),
                        label="",
                        interactive=False,
                        show_row_numbers=True,
                    )
                with gr.Tab(label="base models"):
                    base_table = gr.Dataframe(
                        [i[0] for i in models["base"]],
                        headers=["name"],
                        col_count=(1, "fixed"),
                        label="",
                        interactive=False,
                        show_row_numbers=True,
                    )
                with gr.Tab(label="standalone models"):
                    standalone_table = gr.Dataframe(
                        [[i[0], i[2]] for i in models["standalone"]],
                        headers=["name", "objective"],
                        col_count=(2, "fixed"),
                        label="",
                        interactive=False,
                        show_row_numbers=True,
                    )
                with gr.Tab(label="LoRA models"):
                    lora_tabel = gr.Dataframe(
                        [[i[0], i[2]] for i in models["lora"]],
                        headers=["name", "objective"],
                        col_count=(2, "fixed"),
                        label="",
                        interactive=False,
                        show_row_numbers=True,
                    )
    # ------ user interaction events -------
    btn.click(
        fn=dummy,
        inputs=[
            model_name,
            token_name,
            vocab_fn,
            step,
            batch_size,
            sequence_size,
            method,
            temperature,
            prompt,
            scaffold,
        ],
        outputs=[img, result, chemfig, message],
    )
    btn_refresh.click(
        fn=refresh,
        inputs=[model_name, vocab_fn, token_name],
        outputs=[
            vocab_table,
            base_table,
            standalone_table,
            lora_tabel,
            model_name,
            vocab_fn,
        ],
    )
    token_name.input(
        fn=lambda x, y: gr.Dropdown(
            list(vocabs.keys()), value=y, label="vocabulary", visible=x == "SELFIES"
        ),
        inputs=[token_name, vocab_fn],
        outputs=vocab_fn,
    )
    method.input(
        fn=lambda x, y: gr.Slider(
            0.0,
            2.5,
            y,
            step=0.001,
            label="temperature",
            visible=x == "ODE",
        ),
        inputs=[method, temperature],
        outputs=temperature,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public", default=False, help="open to public", action="store_true"
    )
    args = parser.parse_args()
    app.launch(share=args.public)
