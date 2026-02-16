# FAIR-DA4ER

This repository provides code and scripts for training **Ditto** models for **Entity Resolution (ER)**, with optional support for **FAIR-DA4ER** data augmentation methods.
The project is developed in the context of the FAIR initiative and focuses on **domain-constrained data augmentation** for ER tasks.

## Requirements

- Python 3.9
- CUDA-compatible GPU and drivers (optional, but recommended for training)

## Setup

1. Clone the repository

2. Create and activate a virtual environment (recommended)
   - `python3 -m venv venv`
   - `source venv/bin/activate`

3. Install dependencies
   `pip install -r requirements.txt`

4. Download Ditto stopwords
   `python ditto/download_stopwords.py`

5. Make the training script executable
   `chmod +x train_ditto.sh`

## Data Preparation

Place your dataset inside the `data/` folder using **Ditto’s serialization format**.

Each dataset should consist of pairs of records with an associated label.
Please refer to the original Ditto documentation or paper for details on the expected input format.

### Task Configuration

Training tasks are specified in the `configs.json` file.
Each configuration defines dataset paths and task-specific metadata.

Example `configs.json` entry:

```json
{
  "name": "Example_Dataset",
  "task_type": "classification",
  "vocab": ["0", "1"],
  "trainset": "data/example/train.txt",
  "validset": "data/example/valid.txt",
  "testset": "data/example/test.txt"
}
```

## Training

You can train the model in two ways.

### Option 1 (recommended): Use the provided shell script
`bash train_ditto.sh`

### Option 2: Run the training script directly
`python ditto/train_ditto.py`

For a detailed explanation of training parameters and their meaning, refer to:
- `ditto/train_ditto.py`
- The original **Ditto** paper

## Data Augmentation (Optional)

To enable **FAIR-DA4ER** data augmentation methods, refer to the code and scripts in the `mixup/` folder.

This folder contains experimental techniques for generating **synthetic, domain-constrained samples** aimed at improving Entity Resolution performance, especially in low-data regimes.

### Related Publication
If you are interested in the theoretical background and experimental evaluation of the proposed methods, we can provide the related paper upon request:

Marco Napoleone, Marco Console, Maurizio Lenzerini, Paolo Merialdo, Laura Papi,
Antonella Poggi, Federico Scafoglieri, and Riccardo Torlone.
Domain-constrained Data Augmentation for Entity Resolution.
Accepted at DEARING 2025 – 2nd International Workshop on Data Engineering meets Intelligent Agents
(co-located with ECML PKDD 2025), Porto, Portugal, September 15–19, 2025.
Proceedings to appear in Springer CCIS (indexed by Web of Science).

## Support

If you encounter issues or need help understanding the code or experiments, please contact the authors.

## References

- Li et al., *Deep Entity Matching with Pre-Trained Language Models*
- FAIR Project – **DA4ER**
