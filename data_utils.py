from transformers import AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json


#      ------------------------ Sentiment Analysis  -------------------------------- #

# load tweets-2020-2021-subset dataset
def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]

def create_text_list(dataset_list):
    text_list = ['' for i in range(len(dataset_list))]
    for i in range(len(dataset_list)):
        text_list[i] = dataset_list[i]['text']
    return text_list

dataset_list = load_json("/home/efrath/repos/NLP-final-project/datasets/timelms/data/tweets/tweets-2020-2021-subset-rnd.jl")
text_list = create_text_list(dataset_list)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
# tokenizer = AutoTokenizer.from_pretrained(MODEL)
# config = AutoConfig.from_pretrained(MODEL)
# # PT
# model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# text = "Covid cases are increasing fast!"
# text = preprocess(text)
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# scores = output[0][0].detach().numpy()
# scores = softmax(scores)
# ranking = np.argsort(scores)
# ranking = ranking[::-1]
# for i in range(scores.shape[0]):
#     l = config.id2label[ranking[i]]
#     s = scores[ranking[i]]
#     print(f"{i+1}) {l} {np.round(float(s), 4)}")

# Pipeline example:
sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)
result = sentiment_task("Covid cases are increasing fast!",return_all_scores=True)

sentiment_file = open("twitter_roberta_sentiment_results.txt","a")
for text in text_list:
    result = sentiment_task(text ,return_all_scores=True)
    print(result)
    sentiment_file.write(result)
sentiment_file.close()