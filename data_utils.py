from transformers import AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from datasets import load_dataset

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
    dataset_modified = dataset.map(lambda dict: {"text": "what sentiment the following sentence has: " + dict["text"],
                                                 "label": get_sentiment_task_results(sentiment_task, dict["text"])})
    dataset_modified.to_csv("sentiment_dataset.csv")

def get_yesno_task_results(yesno_task, text):
    result = yesno_task(text, return_all_scores=True)
    # return [[result[0][0]['score'], result[0][1]['score'], result[0][2]['score']], ""]
    return [0, 0, 0, result[0][0]['score'], result[0][1]['score']]

def create_csv_data_yesno_model():
    MODEL = "nc33/yes_no_qna_deberta_model"
    yesno_task = pipeline("text-classification", model=MODEL, tokenizer=MODEL)
    # change dataset:
    dataset = load_dataset("boolq")
    #dataset = load_dataset("parquet", data_files='tweet_sentiment_multilingual-train.parquet')
    dataset = dataset["train"]
    dataset = dataset.remove_columns(["passage"])
    dataset = dataset.select([0, 10])
    dataset_modified = dataset.map(lambda dict: {"question": "Answer yes or no: " + dict["question"], "answer": get_yesno_task_results(yesno_task, dict["question"])})
    dataset_modified.to_csv("yesno_dataset.csv")

create_csv_data_sentiment_model()
create_csv_data_yesno_model()
print("here")
