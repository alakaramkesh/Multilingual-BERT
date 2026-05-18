from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ud_utils import read_yaml


def compute_accuracy(eval_pred):
    # Compute POS accuracy and ignore labels with value -100.
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    mask = labels != -100
    correct = predictions[mask] == labels[mask]
    accuracy = correct.mean() if correct.size > 0 else 0.0

    return {"accuracy": float(accuracy)}


def make_eval_args(params, output_dir):
    # Build a small Trainer configuration for evaluation only.
    return TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=params["batch_size"],
        report_to=[],
    )


def main():
    params = read_yaml("params.yaml")

    processed_dir = Path(params["paths"]["processed_dir"])
    models_dir = Path("models")
    results_dir = Path(params["paths"]["results_dir"])
    reports_dir = Path(params["paths"]["reports_dir"])

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        params["model_checkpoint"],
        clean_up_tokenization_spaces=False,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    rows = []
    languages = list(params["languages"].keys())

    # Evaluate each trained model on every test language.
    for train_lang in tqdm(languages, desc="Evaluating trained models"):
        model_dir = models_dir / train_lang / "best"
        print(f"Loading model trained on {train_lang}: {model_dir}")

        model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        args = make_eval_args(params, str(models_dir / train_lang / "eval_tmp"))

        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            compute_metrics=compute_accuracy,
        )

        for test_lang in languages:
            test_dataset = load_from_disk(str(processed_dir / test_lang / "test"))
            metrics = trainer.evaluate(test_dataset)
            row = {
                "train_lang": train_lang,
                "test_lang": test_lang,
                "accuracy": metrics["eval_accuracy"],
            }
            rows.append(row)
            print(row)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    long_path = results_dir / "transfer_matrix_long.csv"
    df.to_csv(long_path, index=False)

    matrix = df.pivot(index="train_lang", columns="test_lang", values="accuracy")
    matrix_path = Path(params["paths"]["matrix_csv"])
    matrix.to_csv(matrix_path)

    print("\nTransfer matrix")
    print(matrix.round(4).to_string())

    md = "# Transfer matrix\n\n"
    md += "Rows are training languages and columns are test languages. Values are POS tagging accuracy.\n\n"
    md += matrix.round(4).to_markdown()
    md += "\n"
    Path(params["paths"]["matrix_md"]).write_text(md, encoding="utf-8")

    print(f"\nSaved {long_path}")
    print(f"Saved {matrix_path}")
    print(f"Saved {params['paths']['matrix_md']}")


if __name__ == "__main__":
    main()
