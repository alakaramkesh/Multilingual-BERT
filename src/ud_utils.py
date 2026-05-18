import yaml
import conllu
from datasets import Dataset
import json


def read_yaml(path):
    # Read the project config file.
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json(path):
    # Read a JSON file and return it as a Python dictionary.
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_range_id(token_id):
    # Multiword token ids in conllu are tuples like (3, "-", 4).
    return isinstance(token_id, tuple) and len(token_id) == 3 and token_id[1] == "-"


def is_empty_node_id(token_id):
    # Empty nodes in conllu are tuples like (3, ".", 1).
    return isinstance(token_id, tuple) and len(token_id) == 3 and token_id[1] == "."


def clean_token_form(form):
    # Remove internal spaces before mBERT tokenization.
    return str(form).replace(" ", "")


def load_conllu_sentences(filename):
    # Keep all tokens because warm-up checks multiword tokens too.
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()
    for sentence in conllu.parse(data):
        yield sentence


def load_conllu(filename):
    # Load only real syntactic words with integer ids.
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()
    for sentence in conllu.parse(data):
        words = []
        tags = []
        for token in sentence:
            token_id = token["id"]
            if not isinstance(token_id, int):
                continue
            words.append(clean_token_form(token["form"]))
            tags.append(token["upos"])
        yield words, tags


def normalize_ud_sentence(sentence):
    words = []
    tags = []
    mwt_ranges = []
    # First pass: find multiword tokens and build their combined labels.
    for token in sentence:
        token_id = token["id"]
        # Multiword tokens have range ids, for example (3, "-", 4).
        if is_range_id(token_id):
            start = token_id[0]
            end = token_id[2]
            mwt_ranges.append((start, end))
            # Collect the UPOS tags of the grammatical words inside the range.
            mwt_tags = []
            for child in sentence:
                child_id = child["id"]
                if isinstance(child_id, int) and start <= child_id <= end:
                    mwt_tags.append(child["upos"])
            # Keep the surface token, but use a combined grammatical label.
            words.append(clean_token_form(token["form"]))
            tags.append("+".join(mwt_tags))
    # Second pass: add normal tokens that are not part of a multiword token.
    for token in sentence:
        token_id = token["id"]
        # Skip multiword-token rows and other non-standard ids.
        if not isinstance(token_id, int):
            continue
        # Skip grammatical words already represented by a multiword token.
        is_inside_mwt = any(start <= token_id <= end for start, end in mwt_ranges)
        if is_inside_mwt:
            continue
        # Keep ordinary tokens with their original UPOS label.
        words.append(clean_token_form(token["form"]))
        tags.append(token["upos"])
    return words, tags

def align_labels_with_mbert(tokenizer, words, tags, max_length=512):
    # Align each UD word label with the first mBERT subtoken.
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    labels = []
    word_ids = encoding.word_ids()
    # Build one label for every mBERT token, including special and padding tokens.
    for word_id, offset in zip(word_ids, encoding["offset_mapping"]):
        # Special tokens and padding tokens do not correspond to real words.
        if word_id is None:
            labels.append("<pad>")
            continue
        start, end = offset
        # The first subtoken receives the original UD label.
        if start == 0 and end != 0:
            labels.append(tags[word_id])
        # Continuation subtokens are ignored during training and evaluation.
        else:
            labels.append("<pad>")
    # Offset mappings are only needed for alignment, not for model training.
    encoding.pop("offset_mapping")
    # Store the aligned label sequence next to the model inputs.
    encoding["labels"] = labels
    return encoding


def encode_labels(labels, label_to_id):
    # Turn string labels into integers for HuggingFace.
    return [[label_to_id[tag] for tag in sent] for sent in labels]


def make_dataset(conllu_file, tokenizer, label_to_id, max_length=512):
    rows = []
    for sentence in load_conllu_sentences(conllu_file):
        # Apply the UD multiword-token normalization.
        words, tags = normalize_ud_sentence(sentence)
        # Tokenize with mBERT and align labels to subtokens.
        encoded = align_labels_with_mbert(
            tokenizer,
            words,
            tags,
            max_length=max_length
        )
        # Convert string labels into integer ids.
        encoded["labels"] = [label_to_id[tag] for tag in encoded["labels"]]
        # Each row must contain the keys expected by the model.
        rows.append(encoded)
    return Dataset.from_list(rows)