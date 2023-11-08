from transformers import AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from datasets import load_dataset, concatenate_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, TFT5Model

#      ------------------------ Sentiment Analysis  -------------------------------- #

# Preparing first model - twitter-roberta-base-sentiment-latest:


def get_sentiment_task_results(sentiment_task, text):
    result = sentiment_task(text, return_all_scores=True)
    # return [[result[0][0]['score'], result[0][1]['score'], result[0][2]['score']], ""]
    words_arr = [(result[0][0]['label'], result[0][0]['score']), (result[0][1]['label'], result[0][1]['score']),
                 (result[0][2]['label'], result[0][2]['score'])]
    tensor = create_word_prob_encoding(words_arr)
    x = 1
    return tensor

def create_csv_data_sentiment_model():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)
    dataset = load_dataset("parquet", data_files='tweet_sentiment_multilingual-train.parquet')
    dataset = dataset["train"]
    # dataset = dataset.select([0, 10])
    dataset_base = dataset.map(lambda dict: {"text": "what sentiment the following sentence has: " + dict["text"],
                                                 "label": get_sentiment_task_results(sentiment_task, dict["text"]),
                                                 "task": "sentiment"})
    dataset_base = dataset_base.add_column("idx", [i for i in range(len(dataset))])
    dataset_base.to_csv("sentiment_dataset_base.csv")

    dataset_for_model = dataset_base.remove_columns(["task", "idx"])

    df_pandas = pd.DataFrame(dataset_for_model)
    df_pandas.to_csv("sentiment_dataset.csv", index=False)

    # dataset_for_model.to_csv("sentiment_dataset.csv")
    return len(dataset_base)

def get_yesno_task_results(yesno_task, text):
    result = yesno_task(text, return_all_scores=True)
    # return [[result[0][0]['score'], result[0][1]['score'], result[0][2]['score']], ""]
    words_arr = [(result[0][0]['label'], result[0][0]['score']), (result[0][1]['label'], result[0][1]['score'])]
    tensor = create_word_prob_encoding(words_arr)
    return tensor

def create_csv_data_yesno_model(starting_idx):
    MODEL = "nc33/yes_no_qna_deberta_model"
    yesno_task = pipeline("text-classification", model=MODEL, tokenizer=MODEL)
    # change dataset:
    dataset = load_dataset("boolq")
    #dataset = load_dataset("parquet", data_files='tweet_sentiment_multilingual-train.parquet')
    dataset = dataset["train"]
    dataset = dataset.remove_columns(["passage"])
    # dataset = dataset.select([0, 10])
    dataset = dataset.map(lambda dict: {"question": "Answer yes or no: " + dict["question"],
                                                 "answer": get_yesno_task_results(yesno_task, dict["question"]),
                                                 "task": "yesno"})
    dataset = dataset.add_column("idx", [i+starting_idx for i in range(len(dataset))])
    dataset = dataset.rename_column("question", "text")
    dataset = dataset.rename_column("answer", "label")
    dataset.to_csv("yesno_dataset_base.csv")

    dataset_for_model = dataset.remove_columns(["task", "idx"])

    df_pandas = pd.DataFrame(dataset_for_model)
    df_pandas.to_csv("yesno_dataset.csv", index=False)

    # dataset_for_model.to_csv("yesno_dataset.csv")

def merge_datasets(dataset1_filename, dataset2_filename):
    dataset1 = load_dataset("csv", data_files=dataset1_filename)
    dataset2 = load_dataset("csv", data_files=dataset2_filename)
    dataset1 = dataset1["train"]
    dataset2 = dataset2["train"]

    # dataset1 = dataset1.remove_columns("Unnamed: 0")
    # dataset2 = dataset2.remove_columns("Unnamed: 0")

    final_dataset = concatenate_datasets([dataset1, dataset2])
    final_dataset = final_dataset.shuffle()

    # final_dataset = final_dataset.drop(0)
    df_pandas = pd.DataFrame(final_dataset)
    df_pandas.to_csv("final_dataset.csv", header=False, index=False)

    # final_dataset.to_csv("final_dataset.csv")

def find_index_of_word_in_vocabulary(word):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenized_word = tokenizer.tokenize(word)
    word_index = 0
    if tokenized_word[0] in tokenizer.get_vocab():
        word_index = tokenizer.get_vocab()[tokenized_word[0]]
    return word_index

# def create_word_prob_encoding(words_arr):
#     words_tensor = [0 for i in range(32128)]
#     for word, prob in words_arr:
#         word_index = find_index_of_word_in_vocabulary(word)
#         words_tensor[word_index] = prob
#         print(word, ": ", word_index)
#     return words_tensor

def create_word_prob_encoding(words_arr):
    words_tensor_arr = [("", 0) for i in range(len(words_arr))]
    for i in range(len(words_arr)):
        word, prob = words_arr[i]
        word_index = find_index_of_word_in_vocabulary(word)
        words_tensor_arr[i] = (word_index, prob)
        # print(word, ": ", word_index)
    return words_tensor_arr

curr_idx = create_csv_data_sentiment_model()
create_csv_data_yesno_model(curr_idx)
merge_datasets("sentiment_dataset.csv", "yesno_dataset.csv")
# print("here")
