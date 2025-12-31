# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a template for solving ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework. Some parts of the code are missing (or do not follow the most optimal design choices...) and students are required to fill these parts themselves (as well as writing their own models, etc.).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Required Python version: python >=3.12

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How to use

To recreate the best model, run the following command:

```bash
python train.py -cn=train_460
```

To run inference (evaluate the model and save predictions) on your custom dataset, place your dataset directory in the
project root, then run:

```bash
python inference.py -cn=inference_custom_dir custom_dataset.root="<your_dir_name>" dataloader.batch_size=5
```
Decoder: LM (beam search with a pre-trained language model).

After inference, to calculate metrics on text files with targets and predictions, run this command:
```bash
python calc_metrics.py --pred_dir "data/saved/custom_audio_dir/predictions/custom"
```

## Inference with different decoding strategies
Inference will be performed on the LibriSpeech test-clean dataset.

Argmax:
```bash
python inference.py -cn=inference_argmax
```

Vanilla beam search:
```bash
python inference.py -cn=inference_bs
```

LM:

```bash
python inference.py -cn=inference_clean
```
On the first time LM will be downloaded automatically.

## Metrics, %
| Decoder        | test-clean CER | test-clean WER | test-other CER |      test-other WER |
|---------------|---------------:|---------------:|---------------:|--------------------:|
| Argmax        |          13.53 |          34.60 |              – |                   – |
| Beam Search   |          13.30 |          34.05 |              – |                   – |
| Beam + LM     |      **10.90** |      **22.89** |      **31.19** |           **54.45** |

## BPE
To build a BPE tokenizer using the LibriSpeech corpus, run:
```bash
python build_bpe_model.py
```

To train the model with a BPE encoder, run:
```bash
python train.py -cn=train_bpe
```
The default dataset is LibriSpeech train-clean-100, and the number of epochs is only 5, so you may want to change these
settings in the config file.

## Report
Report could be found [here](https://wandb.ai/a-ivlev-87-hse/asr_homework/reports/-ASR---VmlldzoxNTUwNjM1NQ?accessToken=iz8ixllupi9kagux0gntjxdx03o8kpsde7k094w20ztfbs7l51ffjl9a0u86avil).

## Demo notebook
Demo notebook could be find [here (Colab)](https://colab.research.google.com/drive/1wWUUT46tjivrEoB_sKoUUvNxa8EL7Avz?usp=sharing).

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
