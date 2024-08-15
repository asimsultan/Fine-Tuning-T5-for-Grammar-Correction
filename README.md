
# Fine-Tuning T5 for Grammar Correction

Welcome to the Fine-Tuning T5 for Grammar Correction project! This project focuses on fine-tuning the T5 model for correcting grammatical errors in text.

## Introduction

Grammar correction involves detecting and correcting grammatical errors in text. In this project, we leverage T5 to correct grammar using a dataset of incorrect and correct sentence pairs.

## Dataset

For this project, we will use a custom dataset of incorrect and correct sentence pairs. You can create your own dataset and place it in the `data/grammar_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/asimsultan/grammar_correction_t5.git
cd grammar_correction_t5

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes incorrect and correct sentence pairs. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: incorrect and correct.

# To fine-tune the T5 model for grammar correction, run the following command:
python scripts/train.py --data_path data/grammar_data.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/grammar_data.csv
