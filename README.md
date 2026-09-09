# Horn of Africa Inspect Evals

Automated text-processing and evaluation infrastructure for documenting cross-lingual safety behavior under Amharic-English, Somali-English, and other multilingual distribution shifts.

## Technical Feasibility & Compute Architecture

- **Local inference:** The evaluation pipeline is designed around open-weight models in the sub-8B range, including Gemma 2 2B and Qwen 3.5 8B, with quantized configurations where supported by the local hardware and software stack.
- **No recurring API requirement:** The project is designed to prioritize local open-weight inference rather than depending on commercial inference APIs. The requested $450 infrastructure budget is focused on power and connectivity reliability and external storage.
- **Hardware constraints:** Local compute is a practical constraint of the project. Quantization and staged evaluation are used to keep experiments feasible on limited hardware rather than assuming datacenter-scale resources.

## Theory of Change & Downstream Implementation

The project is intended to produce structured evaluation data that can be reused by researchers working on multilingual safety and model evaluation.

1. **Inspect-compatible evaluation workflow:** Evaluation samples and outputs are represented using structured JSONL formats suitable for reproducible evaluation workflows.
2. **Regional linguistic coverage:** The dataset focuses on multilingual and code-switched conditions that are often underrepresented in standard English-centric safety evaluations.
3. **Reproducible analysis:** Preprocessing, dataset schemas, evaluation configuration, and results are separated so that future experiments can be rerun and compared.
4. **Open-source contribution:** Findings can be documented and shared with relevant open-source evaluation and safety communities where appropriate.

## Project Structure

- `scrape_evals.py` — Unicode normalization, multilingual tokenization, script detection, and code-switching detection.
- `inspect_eval.py` — evaluation entry point for the Inspect-based workflow.
- `tests/` — automated tests for the preprocessing pipeline.
- `outputs/` — structured evaluation outputs and sample data.

## Current Status

The repository contains the preprocessing and evaluation infrastructure for the project. Evaluation outputs are recorded separately from the source pipeline so that experimental results can be reproduced and audited.

## Data and Ethics

Only authorized and appropriately sanitized research data should be used. Private messages, credentials, personal information, and other sensitive data should not be committed to this repository.

## Reproducibility

The project separates:

1. data preparation,
2. evaluation configuration,
3. model execution,
4. structured outputs, and
5. subsequent analysis.

This separation is intended to make multilingual safety experiments easier to inspect and reproduce.
