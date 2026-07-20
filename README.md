# TFCL

Official implementation of **TFCL**, a consistency-learning framework for robust speech deepfake detection under acoustic front-end (AFE) distortions.

> **Paper:** *Time-Frequency Consistency Learning for Robust Speech Deepfake Detection*  
> **Authors:** Jun Xue, Zhuolin Yi, Yanzhen Ren, Yihuan Huang, Jiayu Xiong, Yi Chai, Guanxiang Feng, Jiajun Liu, and Tong Zhang  
> **Venue:** ACM Multimedia 2026

## Overview

TFCL improves the robustness of speech deepfake detection by learning consistent representations between clean speech and speech processed by an acoustic front-end (AFE) pipeline.

During training, paired clean and AFE-processed utterances are used for consistency learning. During inference, only a single input utterance is required. Therefore, the consistency-learning branch introduces no additional inference-time computation.

<!--
<p align="center">
  <img src="assets/framework.png" width="95%" alt="TFCL framework">
</p>
-->

## Repository Structure

```text
TFCL/
├── code/
│   ├── main_train.py        # Training program
│   ├── eval.py              # Evaluation program
│   ├── model.py             # TFCL model definition
│   ├── data_utils_SSL.py    # Dataset and data-loading utilities
│   └── ...
├── scripts/
│   ├── train.sh             # Training entry point
│   └── eval.sh              # Evaluation entry point
├── Models/                  # Model checkpoints or checkpoint links
├── protocols/               # Protocol files, if provided
├── README.md
└── LICENSE
```

The exact filenames may differ slightly from the structure above. Please check `scripts/train.sh` and `scripts/eval.sh` for the actual arguments and path settings.

## Environment Setup

The environment configuration follows the SSL anti-spoofing baseline implementation:

- [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)

Please follow the installation instructions in that repository to prepare Python, PyTorch, Fairseq, and the other required dependencies.

A typical setup is:

```bash
git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
cd SSL_Anti-spoofing

conda create -n tfcl python=3.7
conda activate tfcl

# Install the PyTorch and CUDA versions compatible with your system.
# Then install Fairseq and the remaining dependencies according to
# the baseline repository.
```

After preparing the environment, clone this repository:

```bash
git clone https://github.com/JunXue-tech/TFCL.git
cd TFCL
```

## Pre-trained SSL Model

TFCL uses the **XLS-R 300M** model as the self-supervised speech front end.

Download the XLS-R checkpoint by following the instructions in:

- [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)

Then update the XLS-R checkpoint path in `model.py` or the corresponding configuration file.

## Pre-trained TFCL Checkpoint

We provide the best-performing TFCL checkpoint used for evaluation:

- [TFCL_best_ckpt.pth](https://huggingface.co/datasets/JunXueTech/TFCL/blob/main/TFCL_best_ckpt.pth)

The checkpoint can be downloaded using the Hugging Face CLI:

```bash
pip install -U huggingface_hub

hf download JunXueTech/TFCL \
  TFCL_best_ckpt.pth \
  --repo-type dataset \
  --local-dir ./pretrained_models
```

The downloaded checkpoint will be stored at:

```text
./pretrained_models/TFCL_best_ckpt.pth
```

Set the checkpoint path in `scripts/eval.sh`:

```bash
MODEL_PATH="./pretrained_models/TFCL_best_ckpt.pth"
```

The released checkpoint contains the parameters required for single-branch inference, including the SSL encoder and AASIST detection backend.

## Data Preparation

### 1. Clean ASVspoof 2019 data

Download the official ASVspoof 2019 database from:

- [ASVspoof 2019 database — University of Edinburgh DataShare](https://datashare.is.ed.ac.uk/handle/10283/3336)

This project uses the **Logical Access (LA)** partition. Prepare the training, development, and evaluation subsets according to the paths configured in `scripts/train.sh` and `scripts/eval.sh`.

A recommended directory layout is:

```text
/path/to/data/
└── ASVspoof2019/
    ├── ASVspoof2019_LA_train/
    ├── ASVspoof2019_LA_dev/
    ├── ASVspoof2019_LA_eval/
    └── protocols/
```

### 2. AFE-processed data

The AFE-processed datasets used in this project are available at:

- [JunXueTech/TFCL on Hugging Face](https://huggingface.co/datasets/JunXueTech/TFCL)

Download the dataset repository:

```bash
pip install -U huggingface_hub

hf download JunXueTech/TFCL \
  --repo-type dataset \
  --local-dir /path/to/TFCL_data
```

Extract the training/development archive:

```bash
tar -xzf \
  /path/to/TFCL_data/ASVspoof2019_train_dev_data_vad.tar.gz \
  -C /path/to/data/
```

Extract the AFE evaluation archive:

```bash
tar -xzf \
  /path/to/TFCL_data/ASVspoof2019_eval_data_AFE.tar.gz \
  -C /path/to/data/
```

After extraction, update the dataset and protocol paths in `scripts/train.sh` and `scripts/eval.sh`.

## Training

Before training, configure the following items in `scripts/train.sh`:

- clean training and development data paths;
- AFE-processed training and development data paths;
- training and development protocol paths;
- XLS-R checkpoint path;
- output directory;
- GPU device;
- batch size and other training hyperparameters.

Run:

```bash
bash scripts/train.sh
```

Training logs, score files, TensorBoard records, and checkpoints will be written to the output directory configured in the script.

The best checkpoint is selected according to the development-set EER. The saved checkpoint contains only the parameters required for single-branch inference.

## Evaluation

Before evaluation, configure the following items in `scripts/eval.sh`:

- evaluation data root;
- evaluation protocol path;
- TFCL checkpoint path;
- AFE-processing stage or evaluation subset;
- score and result output paths;
- GPU device.

Evaluate the selected clean or AFE-processed subset with:

```bash
bash scripts/eval.sh
```

The evaluation script reports the EER and AUC and saves utterance-level prediction scores to the configured output directory.

To evaluate multiple AFE-processing stages, enable the corresponding evaluation commands in `scripts/eval.sh`.

## Citation

If you find this repository useful for your research, please cite our conference paper:

```bibtex
@inproceedings{xue2026tfcl,
  title     = {Time-Frequency Consistency Learning for Robust Speech Deepfake Detection},
  author    = {Xue, Jun and Yi, Zhuolin and Ren, Yanzhen and Huang, Yihuan and Xiong, Jiayu and Chai, Yi and Feng, Guanxiang and Liu, Jiajun and Zhang, Tong},
  booktitle = {Proceedings of the ACM International Conference on Multimedia},
  year      = {2026}
}
```

The page numbers, DOI, and other publication metadata will be added after they become publicly available.

## Acknowledgements

This implementation is developed based on and inspired by:

- [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)

We thank the authors for releasing their implementation.

## License

Please refer to the [LICENSE](LICENSE) file for the terms of use.

Users should also comply with the licenses and terms of use of the ASVspoof 2019 database, the XLS-R checkpoint, and any third-party code included or adapted in this repository.

## Contact

For questions about the code or data, please open an issue in this repository.
