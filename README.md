# TFCL

Official implementation of **TFCL**, a consistency-learning framework for robust speech deepfake detection under acoustic front-end (AFE) distortions.

> **Paper:** *[Paper title will be added after publication]*  
> **Authors:** *[To be added]*  
> **Venue:** *[To be added]*

## Overview

TFCL improves the robustness of speech deepfake detection by learning consistent representations between clean speech and speech processed by an acoustic front-end (AFE) pipeline.

During training, paired clean and AFE-processed utterances are used for consistency learning. During inference, only a single input utterance is required, so the consistency branch introduces no additional inference-time computation.

<!-- Optional: add the framework figure to assets/framework.png and uncomment below. -->
<!--
<p align="center">
  <img src="assets/framework.png" width="95%" alt="TFCL framework">
</p>
-->

## Repository Structure

```text
TFCL/
├── train.sh                 # Training entry point
├── eval.sh                  # Evaluation entry point for different AFE stages
├── main_train.py            # Training program
├── eval.py                  # Evaluation program
├── model.py                 # TFCL model definition
├── data_utils.py            # Dataset and data-loading utilities
├── protocols/               # Protocol files, if provided
├── configs/                 # Configuration files, if provided
├── README.md
└── LICENSE
```

The exact filenames may differ slightly from the structure above. Please check `train.sh` and `eval.sh` for the actual entry-point arguments and paths.

## Environment Setup

The environment configuration follows the official implementation of the SSL anti-spoofing baseline:

- [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)

Please follow its installation instructions to prepare Python, PyTorch, Fairseq, and the required dependencies.

A typical setup is:

```bash
git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
cd SSL_Anti-spoofing

conda create -n tfcl python=3.7
conda activate tfcl

# Install the PyTorch/CUDA versions compatible with your server.
# Then install the Fairseq version and remaining dependencies
# according to the baseline repository.
```

After preparing the environment, clone this repository:

```bash
git clone https://github.com/JunXue-tech/TFCL.git
cd TFCL
```

## Pre-trained SSL Model

TFCL uses the XLS-R 300M model as the SSL front end. Download the checkpoint following the instructions in:

- [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)

Then update the checkpoint path in the configuration, training script, or model file as required by this implementation.

## Data Preparation

### 1. Clean ASVspoof 2019 data

Download the official ASVspoof 2019 database from:

- [ASVspoof 2019 database — University of Edinburgh DataShare](https://datashare.is.ed.ac.uk/handle/10283/3336)

This project uses the **Logical Access (LA)** partition. Prepare the required training, development, and evaluation subsets according to the paths configured in `train.sh` and `eval.sh`.

A recommended layout is:

```text
/path/to/data/
└── ASVspoof2019/
    ├── ASVspoof2019_LA_train/
    ├── ASVspoof2019_LA_dev/
    ├── ASVspoof2019_LA_eval/
    └── protocols/
```

### 2. AFE-processed data

The AFE-processed datasets used by this project are available at:

- [JunXueTech/TFCL on Hugging Face](https://huggingface.co/datasets/JunXueTech/TFCL)

Download and extract the required archives. For example:

```bash
# Install the Hugging Face CLI if necessary.
pip install -U huggingface_hub

# Download the dataset repository.
hf download JunXueTech/TFCL \
  --repo-type dataset \
  --local-dir /path/to/TFCL_data

# Example extraction commands.
tar -xzf /path/to/TFCL_data/ASVspoof2019_train_dev_data_vad.tar.gz \
  -C /path/to/data/

tar -xzf /path/to/TFCL_data/ASVspoof2019_eval_data_AFE.tar.gz \
  -C /path/to/data/
```

After extraction, modify the dataset and protocol paths in `train.sh` and `eval.sh`.

## Training

Before training, check the following settings in `train.sh`:

- clean training and development data paths;
- AFE-processed training and development data paths;
- protocol paths;
- XLS-R checkpoint path;
- output directory;
- GPU IDs and training hyperparameters.

Run:

```bash
bash train.sh
```

Training outputs, logs, and checkpoints will be written to the output directory configured in the script.

## Evaluation

Before evaluation, configure the following items in `eval.sh`:

- evaluation data root;
- protocol path;
- trained checkpoint path;
- AFE stage or evaluation subset;
- score and result output paths;
- GPU ID.

Evaluate the model on the different AFE-processing stages with:

```bash
bash eval.sh
```

The script should be configured to evaluate the desired AFE stages and report the corresponding detection metrics.

## Citation

The citation information will be added after the paper metadata becomes publicly available.

```bibtex
@article{tfcl,
  title   = {To be added},
  author  = {To be added},
  journal = {To be added},
  year    = {To be added}
}
```

## Acknowledgements

This implementation is developed based on and inspired by the following repository:

- [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)

We thank the authors for releasing their implementation.

## License

Please refer to the `LICENSE` file for the terms of use.

Before selecting a license, make sure it is compatible with all third-party source code included or adapted in this repository.

## Contact

For questions about the code or data, please open a GitHub issue.
