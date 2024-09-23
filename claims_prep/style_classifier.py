import pandas as pd
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import numpy as np
import evaluate
import os
import argparse

"""
Script that trains a binary fine-tuned classifier of tweets vs. non-tweets. 
The following arguments should be passed: 
--tweets_path: path to pkl containing a dataframe of tweets (see style_classifier_data scripts)
--non_tweets_path: path to pkl containing a dataframe of non-tweets (see style_classifier_data scripts)
--model_name: name of HF model to be used for fine-tuning
--tokenizer_name: name of HF tokenizer to be used for tokenizing datasets

"""


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding=True, truncation=True)

def compute_metrics(eval_pred):
    metric = evaluate.load('accuracy')
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')
    f1_metric = evaluate.load('f1')

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels), \
        precision_metric.compute(predictions=predictions, references=labels), \
        recall_metric.compute(predictions=predictions, references=labels), \
        f1_metric.compute(predictions=predictions, references=labels)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--tweets_path", type=str, help="Path for dataset that contains tweets dataframe in pkl format")
    parser.add_argument("--non_tweets_path", type=str, help="Path for dataset that contains non-tweets dataframe in pkl format")
    parser.add_argument("--model_name", type=str, help="Name of model to be loaded from HF")
    parser.add_argument("--tokenizer_name", type=str, help="Name of tokenizer to be loaded from HF")

    args = parser.parse_args()

    tweets_path = args.tweets_path
    non_tweets_path = args.non_tweets_path
    model_name = args.model_name
    tokenizer_name = args.tokenizer_name

    tweets_df = pd.read_pickle(tweets_path)
    non_tweets_df = pd.read_pickle(non_tweets_path)

    classifier_data = pd.concat([tweets_df, non_tweets_df], ignore_index=True)

    labels = [1 if label == 'social_media' else 0 for label in classifier_data['label'].values]
    classifier_data['label'] = labels

    dataset = Dataset.from_pandas(classifier_data)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tokenized_dataset = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True)
    print(len(tokenized_dataset['train']))
    print(len(tokenized_dataset['test']))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    model.to(device)

    training_args = TrainingArguments(
        output_dir="results",
        logging_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        tokenizer=tokenizer
        )

    trainer.train()

    trainer.save_model("results_de/models")

    # Evaluate model
    predictions = trainer.predict(tokenized_dataset["test"])
    accuracy_metric = evaluate.load('accuracy')
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')
    f1_metric = evaluate.load('f1')

    preds = []
    for pred in predictions.predictions:
        pred_0 = pred[0]
        pred_1 = pred[1]
        if pred_1 >= pred_0:
            preds.append(1)
        else:
            preds.append(0)

    accuracy = accuracy_metric.compute(predictions=preds, references=predictions.label_ids)
    print(f'Accuracy: {accuracy}')
    precision = precision_metric.compute(predictions=preds, references=predictions.label_ids)
    print(f'Precision: {precision}')
    recall = recall_metric.compute(predictions=preds, references=predictions.label_ids)
    print(f'Recall: {recall}')
    f1 = f1_metric.compute(predictions=preds, references=predictions.label_ids)
    print(f'F1: {f1}')


if __name__ == "__main__":
    main()
