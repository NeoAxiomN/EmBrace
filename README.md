# [ICML 2026] EmBrace: A Collective Knowledge Fusion Framework Toward Unified EEG Foundation Models


## 🔧 Setup

Requirements:

- Python 3.10+
- A PyTorch installation compatible with your CUDA runtime if you plan to use GPUs

Install from the repository root:

```bash
pip install -e .
```

Check the available commands:

```bash
python -m embrace.main --help
python -m embrace.main finetune --help
python -m embrace.main fuse --help
```

## 📂 Datasets and Checkpoints

This repository does not ship raw datasets or pretrained weights.

- Use `--dataset-path` to point to your processed dataset directory.
- Use `--foundation-dir` for finetuning checkpoints.
- Use `--foundation-dir-cbramod`, `--foundation-dir-labram`, `--foundation-dir-codebrain`, and `--foundation-dir-csbrain` for fusion runs.


## 🚀 Quick Start
EmBrace provides one unified training and evaluation entrypoint:

```bash
python -m embrace.main <subcommand>
```


For EmBrace knowledge fusion:
```bash
python -m embrace.main fuse \
  --dataset bciciv2a \
  --model embrace \
  --dataset-path /path/to/BCICIV2a \
  --target-name cbramod \ # or codebrain, labram
  --source-names labram cbramod codebrain \
  --source-tag la-cb-co \
  --foundation-dir-cbramod /path/to/cbramod_pretrained_weights.pth \
  --foundation-dir-labram /path/to/labram_pretrained_weights.pth \
  --foundation-dir-codebrain /path/to/codebrain_pretrained_weights.pth \
  --device cuda \
  --runtime-mode single
```

For finetune EEG foundation model:

```bash
python -m embrace.main finetune \
  --dataset bciciv2a \
  --model cbramod \ # or codebrain, labram
  --dataset-path /path/to/BCICIV2a \
  --foundation-dir /path/to/cbramod_pretrained_weights.pth
```

For task-specific models:
```bash
python -m embrace.main etms \
  --dataset bciciv2a \
  --model eegnet \ # or deformer, sera, sparc, contra, cnntrans, ffcl, sttrans
  --dataset-path /path/to/BCICIV2a
```

For model selection:

```bash
python -m embrace.main select \
  --dataset bciciv2a \
  --model embrace \
  --dataset-path /path/to/BCICIV2a \
  --source-names labram cbramod codebrain \
  --source-tag la-cb-co
```

Run multi-seed or DDP jobs when needed:

```bash
python -m embrace.main fuse \
  --seeds 0 1 2 3 4 \
  --dataset bciciv2a \
  --model embrace \
  --dataset-path /path/to/BCICIV2a \
  --runtime-mode ddp \ # or cpu, single, ddp
  --runtime-world-size 8 \
  --runtime-master-port 29500
```


## ♻️ Saved Runs

Restart training from a saved resolved config:

```bash
python -m embrace.main rescue \
  --exp-dir outputs/fuse/.../seed_0 \
  --output-base-dir ./tmp
```

Evaluate a saved checkpoint:

```bash
python -m embrace.main evaluate \
  --exp-dir outputs/fuse/.../seed_0
```

Notes:

- `rescue` uses `config_resolved.yaml` as the new run baseline and restarts training from scratch.
- `evaluate` reads the saved config plus a checkpoint file and runs test-only evaluation.

## 🗂 Output Layout

All runs share the same layout:

```text
outputs/<task>/<dataset>/<model_or_key>/<timestamp>/seed_N
```

## 🔗  Citation
If you're using this repository in your research or applications, please cite using the following BibTeX:
```bibtex
@inproceedings{anonymous2026embrace,
  title={EmBrace: A Collective Knowledge Fusion Framework Toward Unified {EEG} Foundation Models},
  author={Anonymous},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
}
```

## 🌹 Acknowledgments 👍
We sincerely appreciate the previous works, including BIOT ([GitHub](https://github.com/ycq091044/BIOT)), LaBraM ([GitHub](https://github.com/935963004/LaBraM)), CodeBrain ([Github](https://github.com/jingyingma01/CodeBrain)), CBraMod ([GitHub](https://github.com/wjq-learning/CBraMod)), and CSBrain ([GitHub](https://github.com/yuchen2199/CSBrain)) for their open-source implementation.

