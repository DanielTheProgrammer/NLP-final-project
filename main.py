from transformers import AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM




#      ------------------------ Sentiment Analysis  -------------------------------- #

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
# sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)
# result = sentiment_task("Covid cases are increasing fast!",return_all_scores=True)
# print(result)

#      ------------------------ Paraphrase Generation: option 1 -------------------------------- #


# tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
# model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
#
# # Define the pipeline
# paraphrase_pipeline = pipeline("text-generation", model=model,tokenizer=tokenizer)
#
#
# # Generate paraphrases using the pipeline
# paraphrases = paraphrase_pipeline(
#     "In this course, we will teach you how to",
#     max_length=30,
#     num_return_sequences=1
# )
# print(paraphrases)

#      ------------------------ Paraphrase Generation: option 2 -------------------------------- #

pipe = pipeline("text2text-generation", model="ramsrigouthamg/t5_sentence_paraphraser")
output = pipe("question: Which is capital city of India? context: New Delhi is India's capital")
print(output)





