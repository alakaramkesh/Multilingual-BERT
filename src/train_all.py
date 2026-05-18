import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
from ud_utils import load_json, read_yaml

def set_seed(seed):
    # fix random seeds to make training more reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_accuracy(eval_pred):
    # Compute POS accuracy while ignoring labels with value -100.
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Only real labels should be evaluated.
    mask = labels != -100
    correct = predictions[mask] == labels[mask]
    accuracy = correct.mean() if correct.size > 0 else 0.0
    return {"accuracy": float(accuracy)}

def make_training_args(params, output_dir):
    # Build the HuggingFace training configuration from params.yaml.
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=params["num_train_epochs"],
        weight_decay=params["weight_decay"],
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[],
        seed=params["seed"],
    )

def main():
    params = read_yaml("params.yaml")
    set_seed(params["seed"])
    label_data = load_json(params["paths"]["label_map"])
    label_to_id = label_data["label_to_id"]
    # Remove <pad> from model labels because -100 is only used in the loss.
    model_label_to_id = {label: idx for label, idx in label_to_id.items() if idx != -100}
    id_to_label = {idx: label for label, idx in model_label_to_id.items()}
    results = []
    tokenizer = AutoTokenizer.from_pretrained(
        params["model_checkpoint"],
        clean_up_tokenization_spaces=False,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    #for lang in params["languages"]:
    for lang in tqdm(params["languages"], desc="Training languages"):
        # Train one POS tagger for each selected language.
        print(f"\nTraining POS tagger on {lang}")
        train_dataset = load_from_disk(os.path.join(params["paths"]["processed_dir"], lang, "train"))
        dev_dataset = load_from_disk(os.path.join(params["paths"]["processed_dir"], lang, "dev"))
        output_dir = os.path.join("models", lang)
        model = AutoModelForTokenClassification.from_pretrained(
            params["model_checkpoint"],
            num_labels=len(model_label_to_id),
            id2label=id_to_label,
            label2id=model_label_to_id,
        )
        training_args = make_training_args(params, output_dir)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_accuracy,
            data_collator=data_collator,
        )
        trainer.train()
        dev_metrics = trainer.evaluate()
        best_dir = os.path.join(output_dir, "best")
        trainer.save_model(best_dir)
        row = {
            "train_language": lang,
            "dev_accuracy": dev_metrics.get("eval_accuracy"),
            "dev_loss": dev_metrics.get("eval_loss"),
        }
        results.append(row)
        print(row)
    os.makedirs(params["paths"]["results_dir"], exist_ok=True)
    results_path = os.path.join(params["paths"]["results_dir"], "dev_results.csv")
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()