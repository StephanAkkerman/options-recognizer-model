# Options Recognizer

Extract option-contract details from tweets: `$AAPL 350 C 11/06/2026 $1.2M 4.37avg` becomes a ticker, strike, option type, expiry, premium and price.

---
<p align="center">
  <img alt="Python versions" src="https://img.shields.io/github/actions/workflow/status/StephanAkkerman/options-recognizer-model/pyversions.yml?label=python%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13&logo=python&style=flat-square">
  <img src="https://img.shields.io/github/license/StephanAkkerman/options-recognizer-model.svg?color=brightgreen" alt="License">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-261230.svg" alt="Code style: ruff"></a>
  <a href="https://huggingface.co/StephanAkkerman/options-recognizer-gliner2-large"><img src="https://img.shields.io/badge/%F0%9F%A4%97-models-yellow?style=flat-square" alt="Models on Hugging Face"></a>
  <a href="https://huggingface.co/datasets/StephanAkkerman/options-ner"><img src="https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow?style=flat-square" alt="Dataset on Hugging Face"></a>
</p>

## Introduction

Traders post option trades and unusual-options-flow alerts on X/Twitter in a terse, slang-heavy format that regexes and general NER models handle poorly. This project fine-tunes [GLiNER2](https://github.com/fastino-ai/GLiNER2) models with LoRA adapters to pull the contract details out of that text, and publishes one ready-to-use model per base size on the Hugging Face Hub, together with the labeled dataset they were trained on.

## Table of Contents 🗂

- [Models](#models)
- [Dataset](#dataset)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Models 🤖

One repo per base model, so you pick a size by picking a repo. All are trained on identical data, splits and seed, and scored with exact-span match on the same held-out test set.

| Model | Base params | Overall F1 |
|---|---|---|
| [options-recognizer-gliner2-large](https://huggingface.co/StephanAkkerman/options-recognizer-gliner2-large) | 486M | **98.7%** |
| [options-recognizer-gliner2-base](https://huggingface.co/StephanAkkerman/options-recognizer-gliner2-base) | 208M | 98.5% |
| [options-recognizer-gliner2.5-small](https://huggingface.co/StephanAkkerman/options-recognizer-gliner2.5-small) | 74M | 86.9% |
| [options-recognizer-gliner2.5-base](https://huggingface.co/StephanAkkerman/options-recognizer-gliner2.5-base) | 194M | 86.8% |

**Best model:** `options-recognizer-gliner2-large` has the highest F1, but `options-recognizer-gliner2-base` is within 0.2 points at less than half the size, so it is the sensible default unless you need every last point. The GLiNER2.5 variants score about 12 points lower on this task. Repos are overwritten when a better adapter is trained, and each version is tagged (`v1`, `v2`, ...).

Per-label F1 of the best model:

| ticker | strike | option_type | expiry | premium | price |
|---|---|---|---|---|---|
| 98.6% | 99.0% | 99.8% | 98.4% | 99.2% | 96.9% |

Each model card has the full precision/recall table, a usage snippet and examples. Without any fine-tuning the same base models reach only 25–64% F1, so the adapters are what make this work.

## Dataset 📚

[`StephanAkkerman/options-ner`](https://huggingface.co/datasets/StephanAkkerman/options-ner) holds the labeled tweets as character spans: 1,287 tweets in `train` and the 227-tweet held-out `test` split that every model was benchmarked on (2,526 test entities). Tweets were scraped, cleaned, pre-labeled with an LLM and corrected by hand in Label Studio.

```python
from datasets import load_dataset

ds = load_dataset("StephanAkkerman/options-ner")
```

## Quick start

```python
import json
import re

from gliner2 import AutoExtractor
from huggingface_hub import snapshot_download

# AlnumBoundarySplitter: copy it from src/core/spans.py or the model card

adapter_dir = snapshot_download("StephanAkkerman/options-recognizer-gliner2-base")
cfg = json.load(open(f"{adapter_dir}/recognizer_config.json"))
model = AutoExtractor.from_pretrained(cfg["base_model"])
model.load_adapter(adapter_dir)
model.set_word_splitter(AlnumBoundarySplitter())  # required

print(model.extract_entities(
    "Unusual flow: $TSLA $300 calls 01/16/2027 sweep, $2.5M premium, filled at 12.40",
    cfg["entity_descriptions"],
    threshold=cfg["threshold"],
))
# {'entities': {'ticker': ['$TSLA'], 'strike': ['$300'], 'option_type': ['calls'],
#               'expiry': ['01/16/2027'], 'premium': ['$2.5M'], 'price': ['12.40']}}
```

Needs `pip install "gliner2>=2" huggingface_hub`. The adapter was trained with a custom word splitter, and without it accuracy drops sharply; the model cards explain why.

## Installation ⚙️

To train, benchmark or publish models yourself, clone the repo and install the requirements:

```bash
git clone https://github.com/StephanAkkerman/options-recognizer-model.git
cd options-recognizer-model
pip install -r requirements.txt
pip install -r requirements-train.txt  # only for training
```

The `data/` folder is not tracked in git; get the labeled data from the [dataset](#dataset).

## Usage ⌨️

Run everything from the repo root with `python -m` (needs `pip install -r requirements.txt`; training also needs `requirements-train.txt`).

| Step | Command |
|---|---|
| Clean scraped tweets (`data/raw/json/o.jsonl` → `data/cleaned/options.jsonl`) | `python -m utils.labeling.clean_data` |
| Publish the raw texts to the Hub | `python -m utils.labeling.clean_data --push user/options-tweets` |
| Pre-label with an LLM | `python -m utils.synthetic.auto_label --interactive` |
| Review in Label Studio | import `data/preds/*.json`, config in `data/label_studio.xml`, export to `data/labeled/` |
| Check for duplicate tasks | `python -m utils.labeling.check_labeled_duplicates` |
| Hold out a test set | `python -m src.maintenance.split_test_set` |
| Train adapters (also splits, benchmarks) | `python -m src.core.train --base-model gliner2-large-v1` (or `all`, or a comma list) |
| Benchmark (exact-span F1 + size/speed, cached) | `python -m src.core.benchmark [--base-model ...] [--all]` |
| Inspect errors / sweep thresholds / compare descriptions | `python -m src.analysis.error_analysis`, `threshold_sweep`, `validate_descriptions` (all take `--base-model`) |
| Publish labeled data (train + held-out test) | `python -m utils.hf.push_dataset_to_hf --repo-id you/options-ner --test-folder data/test --public` |
| Publish the best adapter of every base model | `python -m utils.hf.push_model_to_hf --best --owner you --public` |
| Publish one adapter | `python -m utils.hf.push_model_to_hf models/options_adapter_<base>_vN --owner you --threshold 0.6` |

Pushing model cards runs each adapter on the example texts, which needs `gliner2>=2` (pass `--no-examples` to skip). Run the tests with `python -m pytest`.

### Model sizes

Adapters are trained for each base in `src/core/models.py` (`gliner2.5-small-v1`, `gliner2.5-base-v1`, `gliner2-base-v1`, `gliner2-large-v1`) on identical data, splits and seed, and each is published to its own repo, `<owner>/options-recognizer-<base>` (without the `-v1` suffix). Pick a smaller base for speed or a larger one for accuracy; the benchmark table shows F1 next to base size and ms/doc. Each repo ships a `recognizer_config.json` (base model, labels, descriptions, threshold) so it loads without this codebase.

The entity labels and their GLiNER2 descriptions live in `src/core/labels.py`. GLiNER2 reads the descriptions at train and inference time, so editing them changes model behavior.

## How it works

1. **Collect and clean** tweets about options, then **pre-label** them with an LLM.
2. **Review** the pre-labels by hand in Label Studio and export them to `data/labeled/`.
3. **Hold out** about 15% as a fixed test set; training never sees it.
4. **Train** a LoRA adapter (rank 32, early stopping) per base model, using a word splitter that breaks between letters and digits so spans such as `4.37avg` are word-aligned.
5. **Benchmark** every adapter with exact-span precision/recall/F1 per label; results are cached per test set in `models/benchmark_results.json`.
6. **Publish** the best adapter of each base to its own Hub repo, with a generated model card.

## Project structure

```text
src/core/          labels, base-model registry, word splitter, training, benchmark
src/analysis/      error analysis, threshold sweep, description comparison
src/maintenance/   test-set split
utils/labeling/    cleaning and duplicate checks for the raw tweets
utils/synthetic/   LLM pre-labeling
utils/hf/          publishing the dataset and the model repos
tests/             pytest suite
```

## Citation ✍️

If you use this project in your research, please cite as follows:

```bibtex
@misc{akkerman_options_recognizer,
  author  = {Stephan Akkerman},
  title   = {Options Recognizer},
  year    = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/options-recognizer-model}}
}
```

## Contributing 🛠

Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.\
![https://github.com/StephanAkkerman/options-recognizer-model/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=StephanAkkerman/options-recognizer-model)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
