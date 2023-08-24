import torch
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

# pipe = pipeline("text2text-generation", model="ramsrigouthamg/t5_sentence_paraphraser")
# output = pipe("question: Which is capital city of India? context: New Delhi is India's capital")
# print(output)

#      ------------------------ Paraphrase Generation: option 3 -------------------------------- #

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws", return_dict_in_generate=True)

# gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_ids = tokenizer("Today is a nice day", return_tensors="pt").input_ids

generated_outputs = model.generate(input_ids, do_sample=True, num_return_sequences=3, output_scores=True)

transition_scores = model.compute_transition_scores(generated_outputs.sequences, generated_outputs.scores, normalize_logits=True)

# input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
# encoder-decoder models, like BART or T5.
input_length = 1 if model.config.is_encoder_decoder else input_ids.input_ids.shape[1]
generated_tokens = generated_outputs.sequences[:, input_length:]
for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | logits | probability
    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")


# only use id's that were generated
# gen_sequences has shape [3, 15]
gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]

# let's stack the logits generated at each step to a tensor and transform
# logits to probs
probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)  # -> shape [3, 15, vocab_size]

# now we need to collect the probability of the generated token
# we need to add a dummy dim in the end to make gather work
gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

# now we can do all kinds of things with the probs

# 1) the probs that exactly those sequences are generated again
# those are normally going to be very small
unique_prob_per_sequence = gen_probs.prod(-1)

# print("probs = ", probs)
# print("gen_probs = ", gen_probs)
# print("unique_prob_per_sequence = ", unique_prob_per_sequence)

# 2) normalize the probs over the three sequences
normed_gen_probs = gen_probs / gen_probs.sum(0)
assert normed_gen_probs[:, 0].sum() == 1.0, "probs should be normalized"

# 3) compare normalized probs to each other like in 1)
unique_normed_prob_per_sequence = normed_gen_probs.prod(-1)




