from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
from ud_utils import is_range_id, load_conllu_sentences, normalize_ud_sentence, read_yaml

def main():
    # Load project settings and create output folders.
    params = read_yaml("params.yaml")
    raw_dir = Path(params["paths"]["raw_dir"])
    results_dir = Path(params["paths"]["results_dir"])
    reports_dir = Path(params["paths"]["reports_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Use the French Sequoia test set for the warm-up analysis.
    test_path = raw_dir / "fr" / params["languages"]["fr"]["test"]
    label_counts = Counter()
    mwt_rows = []
    space_rows = []
    for sid, sentence in enumerate(load_conllu_sentences(str(test_path))):
        # Count labels after applying the project UD normalization.
        words, labels = normalize_ud_sentence(sentence)
        label_counts.update(labels)
        for tok in sentence:
            tok_id = tok["id"]
            tok_form = tok["form"]
            # Q4: multiword tokens have range ids such as 3-4 in the original conllu file.
            if is_range_id(tok_id):
                start = tok_id[0]
                end = tok_id[2]
                child_forms = []
                child_labels = []
                for child in sentence:
                    child_id = child["id"]
                    if isinstance(child_id, int) and start <= child_id <= end:
                        child_forms.append(child["form"])
                        child_labels.append(child["upos"])
                mwt_rows.append({
                    "sentence_id": sid,
                    "mwt_id": str(tok_id),
                    "surface_form": tok_form,
                    "grammatical_words": " + ".join(child_forms),
                    "grammatical_labels": " + ".join(child_labels),
                    "combined_label": "+".join(child_labels),
                })
            # Q6: save tokens whose surface form contains spaces.
            if isinstance(tok_form, str) and " " in tok_form:
                space_rows.append({
                    "sentence_id": sid,
                    "id": str(tok_id),
                    "original_form": tok_form,
                    "cleaned_form": tok_form.replace(" ", ""),
                })
    # Q3: save the label distribution.
    label_df = pd.DataFrame(label_counts.items(), columns=["label", "count"]).sort_values("count", ascending=False)
    label_df.to_csv(results_dir / "fr_sequoia_label_distribution.csv", index=False)
    # Q4: save detailed multiword token information.
    mwt_df = pd.DataFrame(mwt_rows)
    mwt_df.to_csv(results_dir / "fr_sequoia_mwt.csv", index=False)
    # Q6: save tokens that contain spaces.
    space_df = pd.DataFrame(space_rows)
    space_df.to_csv(results_dir / "fr_sequoia_tokens_with_spaces.csv", index=False)
    # Q3: save the label distribution plot.
    plt.figure(figsize=(10, 5))
    plt.bar(label_df["label"], label_df["count"])
    plt.xticks(rotation=45, ha="right")
    plt.title("French Sequoia test label distribution")
    plt.tight_layout()
    plt.savefig(results_dir / "fr_sequoia_label_distribution.png")
    plt.close()
    # Print useful outputs directly in the terminal.
    print("Label distribution in the French Sequoia test set")
    print(label_df.to_string(index=False))
    print("Saved files:")
    print(results_dir / "fr_sequoia_label_distribution.csv")
    print(results_dir / "fr_sequoia_label_distribution.png")
    print("\n" + "=" * 80)
    print("Multiword tokens in the French Sequoia test set")
    if mwt_df.empty:
        print("No multiword tokens found.")
    else:
        print(mwt_df.to_string(index=False))
    print("Saved file:")
    print(results_dir / "fr_sequoia_mwt.csv")
    print("\n" + "=" * 80)
    print("Tokens containing spaces")
    if space_df.empty:
        print("No tokens with spaces found.")
    else:
        print(space_df.to_string(index=False))
    print("Saved file:")
    print(results_dir / "fr_sequoia_tokens_with_spaces.csv")

    tokenizer = AutoTokenizer.from_pretrained(
        params["model_checkpoint"],
        clean_up_tokenization_spaces=False,
    )
    example = "Pouvez-vous donner les mêmes garanties au sein de l’Union Européene"
    print(tokenizer.tokenize(example))

if __name__ == "__main__":
    main()
