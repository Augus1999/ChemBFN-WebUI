# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (omozawa SUENO)
"""
Define application behaviours.
"""
import sys
import argparse
from pathlib import Path
from functools import partial
from typing import Tuple, List, Dict, Union

sys.path.append(str(Path(__file__).parent.parent))
from rdkit.Chem import Draw, MolFromSmiles  # type: ignore
from mol2chemfigPy3 import mol2chemfig
import gradio as gr
import torch
from selfies import decoder
from bayesianflow_for_chem import ChemBFN, MLP, EnsembleChemBFN
from bayesianflow_for_chem.data import (
    VOCAB_KEYS,
    FASTA_VOCAB_KEYS,
    load_vocab,
    smiles2vec,
    fasta2vec,
    split_selfies,
)
from bayesianflow_for_chem.tool import (
    sample,
    inpaint,
    optimise,
    adjust_lora_,
    quantise_model_,
)
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

HTML_STYLE = gr.InputHTMLAttributes(
    autocapitalize="off",
    autocorrect="off",
    spellcheck=False,
    autocomplete="off",
    lang="en",
)


def selfies2vec(sel: str, vocab_dict: Dict[str, int]) -> List[int]:
    """
    Tokeniser SELFIES string.

    :param sel: SELFIES string
    :param vocab_dict: vocabulary dictionary
    :type sel: str
    :type vocab_dict: dict
    :return: a list of token indices
    :rtype: list
    """
    s = split_selfies(sel)
    unknown_id = None
    for key, idx in vocab_dict.items():
        if "unknown" in key.lower():
            unknown_id = idx
            break
    return [vocab_dict.get(i, unknown_id) for i in s]


def refresh(
    model_selected: str, vocab_selected: str, tokeniser_selected: str
) -> Tuple[
    List[str], List[str], List[List[str]], List[List[str]], gr.Dropdown, gr.Dropdown
]:
    """
    Refresh model file list.

    :param model_selected: the selected model name
    :param vocab_selected: the selected vocabulary name
    :param tokeniser_selected: the selected tokeniser name
    :type model_selected: str
    :type vocab_selected: str
    :type tokeniser_selected: str
    :return: a list of vocabulary names \n
             a list of base model files \n
             a list of standalone model files \n
             a list of LoRA model files \n
             Gradio Dropdown item \n
             Gradio Dropdown item \n
    :rtype: tuple
    """
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
        filterable=False,
    )
    f = gr.Dropdown(
        list(vocabs.keys()),
        value=vocab_selected,
        label="vocabulary",
        visible=tokeniser_selected == "SELFIES",
        filterable=False,
    )
    return a, b, c, d, e, f


def select_lora(evt: gr.SelectData, prompt: str) -> str:
    """
    Select LoRA model name from Dataframe object.

    :param evt: `~gradio.SelectData` instance
    :param prompt: prompt string
    :type evt: gradio.SelectData
    :type prompt: str
    :return: new prompt string
    :rtype: str
    """
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


def token_name_change_evt(
    token_name: str, vocab_fn: str
) -> Tuple[gr.Dropdown, gr.Tab, gr.Tab]:
    """
    Define token_name-dropdown item change event.

    :param token_name: tokeniser name
    :param vocab_fn: customised vocabulary name
    :type token_name: str
    :type vocab_fn: str
    :return: Dropdown item \n
             Tab item \n
             Tab item \n
    :rtype: tuple
    """
    a = gr.Dropdown(
        list(vocabs.keys()),
        value=vocab_fn,
        label="vocabulary",
        visible=token_name == "SELFIES",
        filterable=False,
    )
    b = gr.Tab(label="LATEX Chemfig", visible=token_name != "FASTA")
    c = gr.Tab(label="gallery", visible=token_name != "FASTA")
    return a, b, c


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
    template: str,
    sar_control: str,
    exclude_token: str,
    quantise: str,
    jited: str,
) -> Tuple[Union[List, None], List[str], str, gr.TextArea, str]:
    """
    Run generation or inpainting.

    :param model_name: model name
    :param token_name: tokeniser name
    :param vocab_fn: customised vocabulary name
    :param step: number of sampling steps
    :param batch_size: batch-size
    :param sequence_size: maximum sequence length
    :param guidance_strength: guidance strength of conditioning
    :param method: `"BFN"` or `"ODE"`
    :param temperature: sampling temperature while ODE-solver used
    :param prompt: prompt string
    :param scaffold: molecular scaffold
    :param template: molecular template
    :param sar_control: semi-autoregressive behaviour flags
    :param exclude_token: unwanted tokens
    :param quantise: `"on"` or `"off"`
    :param jited: `"on"` or `"off"`
    :type model_name: str
    :type token_name: str
    :type vocab_fn: str
    :type step: int
    :type batch_size: int
    :type sequence_size: int
    :type guidance_strength: float
    :type method: str
    :type temperature: float
    :type prompt: str
    :type scaffold: str
    :type template: str
    :type sar_control: str
    :type exclude_token: str
    :type quantise: str
    :type jited: str
    :return: list of images \n
             list of generated molecules \n
             Chemfig code \n
             messages \n
             cache file path
    :rtype: tuple
    """
    _message = []
    base_model_dict = dict(models["base"])
    standalone_model_dict = dict([[i[0], i[1]] for i in models["standalone"]])
    lora_model_dict = dict([[i[0], i[1]] for i in models["lora"]])
    standalone_label_dict = dict([[i[0], i[2] != []] for i in models["standalone"]])
    lora_label_dict = dict([[i[0], i[2] != []] for i in models["lora"]])
    standalone_lmax_dict = dict([[i[0], i[3]] for i in models["standalone"]])
    lora_lmax_dict = dict([[i[0], i[3]] for i in models["lora"]])
    # ------- build tokeniser -------
    if token_name == "SMILES & SAFE":
        vocab_keys = VOCAB_KEYS
        tokeniser = smiles2vec
        trans_fn = lambda x: [i for i in x if (MolFromSmiles(i) and i)]
        img_fn = lambda x: [Draw.MolToImage(MolFromSmiles(i), (500, 500)) for i in x]
        chemfig_fn = lambda x: [mol2chemfig(i, "-r", inline=True) for i in x]
    if token_name == "FASTA":
        vocab_keys = FASTA_VOCAB_KEYS
        tokeniser = fasta2vec
        trans_fn = lambda x: x
        img_fn = lambda _: None  # senseless to provide dumb 2D images
        chemfig_fn = lambda _: [""]  # senseless to provide very long Chemfig code
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
    print("Prompt summary:", prompt_info)  # show prompt info
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
        if quantise == "on":
            quantise_model_(bfn)
        if jited == "on":
            bfn.compile()
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
        if quantise == "on":
            quantise_model_(bfn)
        if jited == "on":
            bfn.compile()
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
        if quantise == "on":
            bfn.quantise()
        if jited == "on":
            bfn.compile()
        _message.append(f"Sequence length is set to {lmax} from model metadata.")
    # ------- inference -------
    allowed_tokens = parse_exclude_token(exclude_token, vocab_keys)
    if not allowed_tokens:
        allowed_tokens = "all"
    scaffold = scaffold.strip()
    template = template.strip()
    if scaffold:
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
        if template:
            _message.append(f"Molecular template {template} ignored.")
    elif template:
        x = [1] + tokeniser(scaffold) + [2]
        x = x + [0 for _ in range(lmax - len(x))]
        x = torch.tensor([x], dtype=torch.long).repeat(batch_size, 1)
        mols = optimise(
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
    else:
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
        gr.TextArea("\n".join(_message), label="message", lines=len(_message)),
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
                filterable=False,
            )
            token_name = gr.Dropdown(
                ["SMILES & SAFE", "SELFIES", "FASTA"],
                label="tokeniser",
                filterable=False,
            )
            vocab_fn = gr.Dropdown(
                list(vocabs.keys()),
                label="vocabulary",
                visible=token_name.value == "SELFIES",
                filterable=False,
            )
            step = gr.Slider(1, 5000, 100, step=1, precision=0, label="step")
            batch_size = gr.Slider(1, 512, 1, step=1, precision=0, label="batch size")
            sequence_size = gr.Slider(
                5, 4096, 50, step=1, precision=0, label="sequence length"
            )
            guidance_strength = gr.Slider(
                0, 25, 4, step=0.05, label="guidance strength"
            )
            method = gr.Dropdown(["BFN", "ODE"], label="method", filterable=False)
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
                prompt = gr.TextArea(
                    label="prompt", lines=12, html_attributes=HTML_STYLE
                )
                scaffold = gr.Textbox(label="scaffold", html_attributes=HTML_STYLE)
                template = gr.Textbox(label="template", html_attributes=HTML_STYLE)
                gr.Markdown("")
                message = gr.TextArea(label="message", lines=2)
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
                with gr.Tab(
                    label="LATEX Chemfig", visible=token_name.value != "FASTA"
                ) as code:
                    chemfig = gr.Code(
                        label="", language="latex", show_line_numbers=True
                    )
            with gr.Tab(
                label="gallery", visible=token_name.value != "FASTA"
            ) as gallery:
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
                sar_control = gr.Textbox(
                    "F",
                    label="semi-autoregressive behaviour",
                    html_attributes=HTML_STYLE,
                )
                gr.Markdown("")
                exclude_token = gr.TextArea(
                    label="exclude tokens",
                    placeholder="key in unwanted tokens separated by comma.",
                    html_attributes=HTML_STYLE,
                )
                quantise = gr.Radio(["on", "off"], value="off", label="quantisation")
                jited = gr.Radio(["on", "off"], value="off", label="JIT")
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
            template,
            sar_control,
            exclude_token,
            quantise,
            jited,
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
        fn=token_name_change_evt,
        inputs=[token_name, vocab_fn],
        outputs=[vocab_fn, code, gallery],
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
    """
    Main function.

    :return:
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="A web-based visualisation tool for ChemBFN method.",
        epilog=f"ChemBFN WebUI {__version__}, developed in Hiroshima University by chemists for chemists. "
        "Visit https://augus1999.github.io/bayesian-flow-network-for-chemistry/ for more details.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--public", default=False, help="open to public", action="store_true"
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)
    args = parser.parse_args()
    app.launch(share=args.public, allowed_paths=[cache_dir.absolute().__str__()])


if __name__ == "__main__":
    main()
