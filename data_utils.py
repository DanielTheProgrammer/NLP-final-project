from transformers import AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from datasets import load_dataset, concatenate_datasets

#      ------------------------ Sentiment Analysis  -------------------------------- #

# Preparing first model - twitter-roberta-base-sentiment-latest:


def get_sentiment_task_results(sentiment_task, text):
    result = sentiment_task(text, return_all_scores=True)
    # return [[result[0][0]['score'], result[0][1]['score'], result[0][2]['score']], ""]
    return [result[0][0]['score'], result[0][1]['score'], result[0][2]['score'], 0, 0]

def create_csv_data_sentiment_model():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)
    dataset = load_dataset("parquet", data_files='tweet_sentiment_multilingual-train.parquet')
    dataset = dataset["train"]
    dataset = dataset.select([0, 10])
    dataset = dataset.map(lambda dict: {"text": "what sentiment the following sentence has: " + dict["text"],
                                                 "label": get_sentiment_task_results(sentiment_task, dict["text"]),
                                                 "task": "sentiment"})
    dataset = dataset.add_column("idx", [i for i in range(len(dataset))])
    dataset.to_csv("sentiment_dataset.csv")
    return len(dataset)

def get_yesno_task_results(yesno_task, text):
    result = yesno_task(text, return_all_scores=True)
    # return [[result[0][0]['score'], result[0][1]['score'], result[0][2]['score']], ""]
    return [0, 0, 0, result[0][0]['score'], result[0][1]['score']]

def create_csv_data_yesno_model(starting_idx):
    MODEL = "nc33/yes_no_qna_deberta_model"
    yesno_task = pipeline("text-classification", model=MODEL, tokenizer=MODEL)
    # change dataset:
    dataset = load_dataset("boolq")
    #dataset = load_dataset("parquet", data_files='tweet_sentiment_multilingual-train.parquet')
    dataset = dataset["train"]
    dataset = dataset.remove_columns(["passage"])
    dataset = dataset.select([0, 10])
    dataset = dataset.map(lambda dict: {"question": "Answer yes or no: " + dict["question"],
                                                 "answer": get_yesno_task_results(yesno_task, dict["question"]),
                                                 "task": "yesno"})
    dataset = dataset.add_column("idx", [i+starting_idx for i in range(len(dataset))])
    dataset = dataset.rename_column("question", "text")
    dataset = dataset.rename_column("answer", "label")
    dataset.to_csv("yesno_dataset.csv")

def merge_datasets(dataset1_filename, dataset2_filename):
    dataset1 = load_dataset("csv", data_files=dataset1_filename)
    dataset2 = load_dataset("csv", data_files=dataset2_filename)
    dataset1 = dataset1["train"]
    dataset2 = dataset2["train"]
    final_dataset = concatenate_datasets([dataset1, dataset2])
    final_dataset = final_dataset.shuffle()
    final_dataset.to_csv("final_dataset.csv")

curr_idx = create_csv_data_sentiment_model()
create_csv_data_yesno_model(curr_idx)
merge_datasets("sentiment_dataset.csv", "yesno_dataset.csv")
print("here")
