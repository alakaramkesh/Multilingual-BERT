from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

from ud_utils import (
    read_yaml,
    load_json,
    load_conllu_sentences,
    normalize_ud_sentence,
    align_labels_with_mbert,
)

def show_q10_q11_check(params, tokenizer):
    # Print one short example to check normalization and mBERT alignment.
    test_path = Path(params["paths"]["raw_dir"]) / "fr" / params["languages"]["fr"]["test"]
    for sentence in load_conllu_sentences(str(test_path)):
        words, tags = normalize_ud_sentence(sentence)
        if any("+" in tag for tag in tags):
            encoding = align_labels_with_mbert(tokenizer, words, tags, max_length=35)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
            print("\nQ10 check: normalized UD words and labels")
            print(pd.DataFrame({"word": words, "label": tags}).head(8).to_string(index=False))
            print("\nQ11 check: mBERT tokens and aligned labels")
            print(pd.DataFrame({"token": tokens, "label": encoding["labels"]}).head(20).to_string(index=False))
            return

def show_q12(params):
    # Print truncation statistics.
    path = Path(params["paths"]["truncation_stats"])
    df = pd.read_csv(path)
    total_sentences = df["sentences"].sum()
    total_truncated = df["truncated_sentences"].sum()
    overall_rate = total_truncated / total_sentences if total_sentences else 0
    print("\nQ12: truncation statistics")
    print(df.to_string(index=False))
    print("\nOverall truncation:")
    print(f"Total sentences: {total_sentences}")
    print(f"Truncated sentences: {total_truncated}")
    print(f"Overall truncation rate: {overall_rate:.6f}")

def show_q13(params):
    # Print the label-to-id mapping.
    label_data = load_json(params["paths"]["label_map"])
    label_to_id = label_data["label_to_id"]
    df = pd.DataFrame(
        [{"label": label, "id": idx} for label, idx in label_to_id.items()]
    ).sort_values("id")
    print("\nQ13: label encoding")
    print(df.to_string(index=False))
    print(f"\n<pad> label id: {label_to_id['<pad>']}")


def main():
    # Print small checks used for the report answers.
    params = read_yaml("params.yaml")
    tokenizer = AutoTokenizer.from_pretrained(
        params["model_checkpoint"],
        clean_up_tokenization_spaces=False,
    )
    show_q10_q11_check(params, tokenizer)
    show_q12(params)
    show_q13(params)


if __name__ == "__main__":
    main()