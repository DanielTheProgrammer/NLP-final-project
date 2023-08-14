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
    return [result[0][0]['score'], result[0][1]['score'], result[0][2]['score']]

def create_csv_data_sentiment_model():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)
    dataset = load_dataset("parquet", data_files='tweet_sentiment_multilingual-train.parquet')
    dataset = dataset["train"]
    dataset_modified = dataset.map(lambda dict: {"text": dict["text"], "label": get_sentiment_task_results(sentiment_task, dict["text"])})
    dataset_modified.to_csv("sentiment_dataset.csv")
