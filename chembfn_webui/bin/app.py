# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
Define application behaviours.
"""
import sys
import argparse
from pathlib import Path
from functools import partial
from typing import Tuple, List, Dict

sys.path.append(str(Path(__file__).parent.parent))
from rdkit.Chem import Draw, MolFromSmiles, MolFromFASTA
from mol2chemfigPy3 import mol2chemfig
import gradio as gr
import torch
from selfies import decoder
from bayesianflow_for_chem import ChemBFN, MLP, EnsembleChemBFN
from bayesianflow_for_chem.data import (
    VOCAB_KEYS,
    AA_VOCAB_KEYS,
    load_vocab,
    smiles2vec,
    aa2vec,
    split_selfies,
)
from bayesianflow_for_chem.tool import sample, inpaint, adjust_lora_
from lib.utilities import (
    find_model,
    find_vocab,
    parse_prompt,
    parse_exclude_token,
    parse_sar_control,
)
from lib.version import __version__

vocabs = find_vocab()
models = find_model()
lora_selected = False  # lora select flag
cache_dir = Path(__file__).parent.parent / "cache"


def selfies2vec(sel: str, vocab_dict: Dict[str, int]) -> List[int]:
    s = split_selfies(sel)
    unknown_id = None
    for key, idx in vocab_dict.items():
        if "unknown" in key.lower():
            unknown_id = idx
            break
    return [vocab_dict.get(i, default=unknown_id) for i in s]


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


def select_lora(evt: gr.SelectData, prompt: str) -> str:
    global lora_selected
    if lora_selected:  # avoid double select
        lora_selected = False
        return prompt
    selected_lora = evt.value
    lora_selected = True
    if evt.index[1] != 0:
        return prompt
    if not prompt:
        return f"<{selected_lora}:1>"
    return f"{prompt};\n<{selected_lora}:1>"


def run(
    model_name: str,
    token_name: str,
    vocab_fn: str,
    step: int,
    batch_size: int,
    sequence_size: int,
    guidance_strength: float,
    method: str,
    temperature: float,
    prompt: str,
    scaffold: str,
    sar_control: str,
    exclude_token: str,
) -> Tuple[List, List[str], str, str, str]:
    _message = []
    base_model_dict = dict(models["base"])
    standalone_model_dict = dict([[i[0], i[1]] for i in models["standalone"]])
    lora_model_dict = dict([[i[0], i[1]] for i in models["lora"]])
    standalone_label_dict = dict([[i[0], i[2] != []] for i in models["standalone"]])
    lora_label_dict = dict([[i[0], i[2] != []] for i in models["lora"]])
    standalone_lmax_dict = dict([[i[0], i[3]] for i in models["standalone"]])
    lora_lmax_dict = dict([[i[0], i[3]] for i in models["lora"]])
    if token_name == "SMILES & SAFE":
        vocab_keys = VOCAB_KEYS
        tokeniser = smiles2vec
        trans_fn = lambda x: [i for i in x if (MolFromSmiles(i) and i)]
        img_fn = lambda x: [Draw.MolToImage(MolFromSmiles(i), (500, 500)) for i in x]
        chemfig_fn = lambda x: [mol2chemfig(i, "-r", inline=True) for i in x]
    if token_name == "FASTA":
        vocab_keys = AA_VOCAB_KEYS
        tokeniser = aa2vec
        trans_fn = lambda x: x
        img_fn = lambda x: [Draw.MolToImage(MolFromFASTA(i), (500, 500)) for i in x]
        chemfig_fn = lambda x: ["null" for _ in x]
    if token_name == "SELFIES":
        vocab_data = load_vocab(vocabs[vocab_fn])
        vocab_keys = vocab_data["vocab_keys"]
        vocab_dict = vocab_data["vocab_dict"]
        tokeniser = partial(selfies2vec, vocab_dict=vocab_dict)
        trans_fn = lambda x: x
        img_fn = lambda x: [
            Draw.MolToImage(MolFromSmiles(decoder(i)), (500, 500)) for i in x
        ]
        chemfig_fn = lambda x: [mol2chemfig(decoder(i), "-r", inline=True) for i in x]
    _method = "bfn" if method == "BFN" else f"ode:{temperature}"
    # ------- build model -------
    prompt_info = parse_prompt(prompt)
    sar_flag = parse_sar_control(sar_control)
    print(prompt_info)
    if not prompt_info["lora"]:
        if model_name in base_model_dict:
            lmax = sequence_size
            bfn = ChemBFN.from_checkpoint(base_model_dict[model_name])
            y = None
            if prompt_info["objective"]:
                _message.append("Objective values ignored by base model.")
        else:
            lmax = standalone_lmax_dict[model_name]
            bfn = ChemBFN.from_checkpoint(
                standalone_model_dict[model_name] / "model.pt"
            )
            if prompt_info["objective"]:
                if not standalone_label_dict[model_name]:
                    y = None
                    _message.append("Objective values ignored.")
                else:
                    mlp = MLP.from_checkpoint(
                        standalone_model_dict[model_name] / "mlp.pt"
                    )
                    y = torch.tensor([prompt_info["objective"]], dtype=torch.float32)
                    y = mlp.forward(y)
            else:
                y = None
            _message.append(f"Sequence length is set to {lmax} from model metadata.")
        bfn.semi_autoregressive = sar_flag[0]
    elif len(prompt_info["lora"]) == 1:
        lmax = lora_lmax_dict[prompt_info["lora"][0]]
        if model_name in base_model_dict:
            bfn = ChemBFN.from_checkpoint(
                base_model_dict[model_name],
                lora_model_dict[prompt_info["lora"][0]] / "lora.pt",
            )
        else:
            bfn = ChemBFN.from_checkpoint(
                standalone_model_dict[model_name] / "model.pt",
                lora_model_dict[prompt_info["lora"][0]] / "lora.pt",
            )
        if prompt_info["objective"]:
            if not lora_label_dict[prompt_info["lora"][0]]:
                y = None
                _message.append("Objective values ignored.")
            else:
                mlp = MLP.from_checkpoint(
                    lora_model_dict[prompt_info["lora"][0]] / "mlp.pt"
                )
                y = torch.tensor([prompt_info["objective"]], dtype=torch.float32)
                y = mlp.forward(y)
        else:
            y = None
        if prompt_info["lora_scaling"][0] != 1.0:
            adjust_lora_(bfn, prompt_info["lora_scaling"][0])
        _message.append(f"Sequence length is set to {lmax} from model metadata.")
        bfn.semi_autoregressive = sar_flag[0]
    else:
        lmax = max([lora_lmax_dict[i] for i in prompt_info["lora"]])
        if model_name in base_model_dict:
            base_model_dir = base_model_dict[model_name]
        else:
            base_model_dir = standalone_model_dict[model_name] / "model.pt"
            lmax = max([lmax, standalone_lmax_dict[model_name]])
        lora_dir = [lora_model_dict[i] / "lora.pt" for i in prompt_info["lora"]]
        mlps = [
            MLP.from_checkpoint(lora_model_dict[i] / "mlp.pt")
            for i in prompt_info["lora"]
        ]
        weights = prompt_info["lora_scaling"]
        if len(sar_flag) == 1:
            sar_flag = [sar_flag[0] for _ in range(len(weights))]
        bfn = EnsembleChemBFN(base_model_dir, lora_dir, mlps, weights)
        y = [torch.tensor([i], dtype=torch.float32) for i in prompt_info["objective"]]
        _message.append(f"Sequence length is set to {lmax} from model metadata.")
    # ------- inference -------
    allowed_tokens = parse_exclude_token(exclude_token, vocab_keys)
    if not allowed_tokens:
        allowed_tokens = "all"
    scaffold = scaffold.strip()
    if not scaffold:
        mols = sample(
            bfn,
            batch_size,
            lmax,
            step,
            y,
            guidance_strength,
            vocab_keys=vocab_keys,
            method=_method,
            allowed_tokens=allowed_tokens,
        )
        mols = trans_fn(mols)
        imgs = img_fn(mols)
        chemfigs = chemfig_fn(mols)
    else:
        x = [1] + tokeniser(scaffold)
        x = x + [0 for _ in range(lmax - len(x))]
        x = torch.tensor([x], dtype=torch.long).repeat(batch_size, 1)
        mols = inpaint(
            bfn,
            x,
            step,
            y,
            guidance_strength,
            vocab_keys=vocab_keys,
            method=_method,
            allowed_tokens=allowed_tokens,
        )
        mols = trans_fn(mols)
        imgs = img_fn(mols)
        chemfigs = chemfig_fn(mols)
    n_mol = len(mols)
    with open(cache_dir / "results.csv", "w", encoding="utf-8", newline="") as rf:
        rf.write("\n".join(mols))
    _message.append(
        f"{n_mol} smaples generated and saved to cache that can be downloaded."
    )
    return (
        imgs,
        mols,
        "\n\n".join(chemfigs),
        "\n".join(_message),
        str(cache_dir / "results.csv"),
    )


with gr.Blocks(title="ChemBFN WebUI") as app:
    gr.Markdown("### WebUI to generate and visualise molecules for ChemBFN method.")
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
            step = gr.Slider(1, 5000, 100, step=1, precision=0, label="step")
            batch_size = gr.Slider(1, 512, 1, step=1, precision=0, label="batch size")
            sequence_size = gr.Slider(
                5, 4096, 50, step=1, precision=0, label="sequence length"
            )
            guidance_strength = gr.Slider(
                0, 25, 4, step=0.05, label="guidance strength"
            )
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
                    btn_download = gr.File(label="download", visible=False)
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
            with gr.Tab(label="advanced control"):
                sar_control = gr.Textbox("F", label="semi-autoregressive behaviour")
                gr.Markdown("")
                exclude_token = gr.TextArea(
                    label="exclude tokens",
                    placeholder="key in unwanted tokens separated by comma.",
                )
    # ------ user interaction events -------
    btn.click(
        fn=run,
        inputs=[
            model_name,
            token_name,
            vocab_fn,
            step,
            batch_size,
            sequence_size,
            guidance_strength,
            method,
            temperature,
            prompt,
            scaffold,
            sar_control,
            exclude_token,
        ],
        outputs=[img, result, chemfig, message, btn_download],
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
    lora_tabel.select(fn=select_lora, inputs=prompt, outputs=prompt)
    result.change(
        fn=lambda x: gr.File(x, label="download", visible=True),
        inputs=btn_download,
        outputs=btn_download,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public", default=False, help="open to public", action="store_true"
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)
    args = parser.parse_args()
    app.launch(share=args.public)


if __name__ == "__main__":
    main()
