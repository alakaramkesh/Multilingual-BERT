import os
import json
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from ud_utils import read_yaml, load_conllu_sentences, normalize_ud_sentence, make_dataset


def collect_all_labels(params):
    # collect labels from all languages and all splits, so no label is missing later.
    labels = {"<pad>"}
    for lang, info in params["languages"].items():
        for split in ["train", "dev", "test"]:
            path = Path(params["paths"]["raw_dir"]) / lang / info[split]
            for sentence in load_conllu_sentences(path):
                _, sent_tags = normalize_ud_sentence(sentence)
                labels.update(sent_tags)

    return sorted(labels)


def save_label_map(labels, output_file):
    # use -100 for <pad> because PyTorch ignores it in token classification loss.
    label_to_id = {label: i for i, label in enumerate(labels) if label != "<pad>"}
    label_to_id["<pad>"] = -100
    id_to_label = {str(i): label for label, i in label_to_id.items() if i != -100}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2, ensure_ascii=False)
    return label_to_id


def compute_corpus_stats(params):
    # I count sentences and normalized UD tokens for each language and split.
    rows = []

    for lang, info in params["languages"].items():
        for split in ["train", "dev", "test"]:
            path = Path(params["paths"]["raw_dir"]) / lang / info[split]
            num_sentences = 0
            num_tokens = 0

            for sentence in load_conllu_sentences(path):
                words, _ = normalize_ud_sentence(sentence)
                num_sentences += 1
                num_tokens += len(words)

            rows.append({
                "language": lang,
                "language_name": info["name"],
                "family": info["family"],
                "script": info["script"],
                "split": split,
                "sentences": num_sentences,
                "tokens": num_tokens,
            })

    return pd.DataFrame(rows)


def compute_truncation_stats(params, tokenizer):
    # I check how often mBERT tokenization goes beyond max_length.
    rows = []

    for lang, info in params["languages"].items():
        for split in ["train", "dev", "test"]:
            path = Path(params["paths"]["raw_dir"]) / lang / info[split]
            num_sentences = 0
            num_truncated = 0
            max_subtokens = 0

            for sentence in load_conllu_sentences(path):
                words, _ = normalize_ud_sentence(sentence)
                encoding = tokenizer(words, is_split_into_words=True, truncation=False)
                length = len(encoding["input_ids"])

                num_sentences += 1
                max_subtokens = max(max_subtokens, length)

                if length > params["max_length"]:
                    num_truncated += 1

            rows.append({
                "language": lang,
                "split": split,
                "sentences": num_sentences,
                "truncated_sentences": num_truncated,
                "truncation_rate": num_truncated / num_sentences if num_sentences else 0,
                "max_subtokens": max_subtokens,
            })

    return pd.DataFrame(rows)



def main():
    params = read_yaml("params.yaml")
    tokenizer = AutoTokenizer.from_pretrained(params["model_checkpoint"],clean_up_tokenization_spaces=False)

    os.makedirs(params["paths"]["processed_dir"], exist_ok=True)
    os.makedirs(params["paths"]["results_dir"], exist_ok=True)

    labels = collect_all_labels(params)
    label_to_id = save_label_map(labels, params["paths"]["label_map"])

    corpus_stats = compute_corpus_stats(params)
    corpus_stats.to_csv(params["paths"]["corpus_stats"], index=False)

    truncation_stats = compute_truncation_stats(params, tokenizer)
    truncation_stats.to_csv(params["paths"]["truncation_stats"], index=False)

    for lang, info in params["languages"].items():
        for split in ["train", "dev", "test"]:
            path = Path(params["paths"]["raw_dir"]) / lang / info[split]
            output_dir = Path(params["paths"]["processed_dir"]) / lang / split

            ds = make_dataset(str(path), tokenizer, label_to_id, max_length=params["max_length"])
            ds.save_to_disk(str(output_dir))

            print(f"Saved dataset {lang}/{split}: {len(ds)} rows")

    print(f"Saved {params['paths']['corpus_stats']}")
    print(f"Saved {params['paths']['truncation_stats']}")


if __name__ == "__main__":
    main()