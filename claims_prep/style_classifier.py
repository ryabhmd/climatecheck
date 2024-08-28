import pandas as pd
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import numpy as np
import evaluate
import wandb
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
wandb.init(project="style-classifier", entity="raya-abu-ahmad")


def tokenize_function(examples):
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

    tweets_df = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/tweets_df.pkl')
    non_tweets_df = pd.read_pickle('/netscratch/abu/Shared-Tasks/ClimateCheck/non_tweets_df.pkl')

    classifier_data = pd.concat([tweets_df, non_tweets_df], ignore_index=True)

    labels = [1 if label == 'tweet' else 0 for label in classifier_data['label'].values]
    classifier_data['label'] = labels

    dataset = Dataset.from_pandas(classifier_data)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True)
    print(len(tokenized_dataset['train']))
    print(len(tokenized_dataset['test']))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)

    model.to(device)

    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = "pairwise-text-classification"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    training_args = TrainingArguments(
        output_dir="results",
        report_to="wandb",
        logging_steps=100,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],tokenizer=tokenizer
        )

    trainer.train()

    trainer.save_model("results/models")

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
