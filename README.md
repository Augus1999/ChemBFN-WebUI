## ChemBFN WebUI: WebUI to generate and visualise molecules for ChemBFN method

<p align="left">
<img src="image/screenshot_0.jpeg" alt="screenshot 0" width="400" height="auto">
<img src="image/screenshot_1.jpeg" alt="screenshot 1" width="400" height="auto">
<img src="image/screenshot_2.jpeg" alt="screenshot 2" width="400" height="auto">
<img src="image/screenshot_3.jpeg" alt="screenshot 3" width="400" height="auto">
<img src="image/screenshot_4.jpeg" alt="screenshot 4" width="400" height="auto">
</p>

### 1. Install

```bash
$ pip install -U chembfn_webui
```

### 2. Place model files

* Place customised vocabulary files (plain text .txt format) under [`chembfn_webui/model/vocab`](./chembfn_webui/model/vocab).
* Place base model files (pretrained model in .pt format) under [`chembfn_webui/model/base_model`](./chembfn_webui/model/base_model).
* Group standalone model files (`model.pt`, optional `mlp.pt`) and configuration file (`config.json`) in one folder for each model and place these folders under [`chembfn_webui/model/standalone_model`](./chembfn_webui/model/standalone_model).
* Group LoRA model files (`lora.pt`, optional `mlp.pt`) and configuration file (`config.json`) in one folder for each model and place these folders under [`chembfn_webui/model/lora`](./chembfn_webui/model/lora).

For example,

```
├───chembfn_webui
    ├───bin
    ├───cache
    ├───lib
    └───model
        ├───base_model
        |   └───zinc15_190m.pt
        ├───lora
        │   └───csd_ees
        │       ├───lora.pt
        |       ├───mlp.pt
        |       └───config.json
        ├───standalone_model
        │   ├───guacamol
        │   |   ├───model.pt
        │   |   └───config.json
        │   └───qm9
        |        ├───model.pt
        |        ├───mlp.pt
        |        └───config.json
        └───vocab
            └───moses_selfies_vocab.txt
```

If placed correctly, all these files can be seen in the "model explorer" tab.

> You can use an external folder to host the models if it follows the same structure as [`chembfn_webui/model`](./chembfn_webui/model). See the next section for the method.

### 3. Launch the program

I. launch the web-UI
```bash
$ chembfn
```

II. launch the web in a public link
```bash
$ chembfn --public
```

III. use an external directory to hold the model files (Linux and MacOS)
```bash
$ export CHEMBFN_WEBUI_MODEL_DIR={YOUR/MODEL/DIR}
$ chembfn
```

IV. use an external directory to hold the model files (Windows)
```bash
$ set CHEMBFN_WEBUI_MODEL_DIR={YOUR/MODEL/DIR}
$ chembfn
```

V. use an external directory to hold the model files (Notebook, Google Colab)
```python
import os
os.environ["CHEMBFN_WEBUI_MODEL_DIR"] = "{YOUR/MODEL/DIR}"
!chembfn --public
```

### 4. Write the prompt

* Leave prompt blank for unconditional generation.
* For standalone models, key in objective values in the format of `[a,b,c,...]` to pass the values to the model.
* Key in `<name:A>` or `<name:A>:[a,b,c,...]` to select LoRA parameter and pass the objective values if necessary, where `name` is the LoRA model name and `A` is the LoRA scaling. You can easily select a LoRA model by clicking the model name in "LoRA models" tab as well.
* You can stack several LoRA models together to form an ensemble model by prompt like `<name1:A1>:[a1,b1,c1,...];<name2:A2>:[a2,b2,...];...`. Note that here `A1`, `A2`, _etc_ are contributions of each model to the ensemble.

### 5. Advanced control

Under "advanced control" tab

* You can control semi-autoregressive behaviours by key in `F` for switching off SAR, `T` for switching on SAR, and prompt like `F,F,T,...` to individually control the SAR in an ensemble model.
* You can add unwanted tokens, e.g., `[Cu],p,[Si]`.

### 6. Generate molecules

Click "RUN" then here you go! If error occured, please check your prompts and settings.

## Where to obtain the models?

* Pretrained models: [https://huggingface.co/suenoomozawa/ChemBFN](https://huggingface.co/suenoomozawa/ChemBFN)
* ChemBFN source code: [https://github.com/Augus1999/bayesian-flow-network-for-chemistry](https://github.com/Augus1999/bayesian-flow-network-for-chemistry)
* ChemBFN document: [https://augus1999.github.io/bayesian-flow-network-for-chemistry/](https://augus1999.github.io/bayesian-flow-network-for-chemistry/)
* ChemBFN package: [https://pypi.org/project/bayesianflow-for-chem/](https://pypi.org/project/bayesianflow-for-chem/)
