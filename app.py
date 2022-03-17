from sre_parse import FLAGS
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')
import re
import yake
language = "en"
max_ngram_size = 2
deduplication_threshold = 0.2
numOfKeywords = 8
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import re
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
from flask import Flask, jsonify, request
print ("module %s loaded" % module_url)

app = Flask(__name__)

def grammer_marks(text):
  allowed_errors = len(re.split(r'[.!?]+', text))
  errors = len(tool.check(text))
  if errors>allowed_errors:
    return 0
  return 1

def embed(input):
  return model(input)

def semantic_score(model_ans,user_ans):
  return np.inner(embed([user_ans]),embed([model_ans]))[0][0]

def keyword_marks(model_ans,user_ans):
  model_ans_keywords = custom_kw_extractor.extract_keywords(model_ans)
  model_keywords = []
  for kw in model_ans_keywords:
    if 1-kw[1]>0.85:
      model_keywords.append(kw[0])
  if len(model_keywords)==0:
    return 1
  user_ans_keywords = custom_kw_extractor.extract_keywords(user_ans)
  user_keywords = []
  for kw in user_ans_keywords:
    if 1-kw[1]>0.85:
      user_keywords.append(kw[0])
  ratio = len(set(model_keywords).intersection(set(user_keywords)))/len(model_keywords)
  return ratio

def get_total(user_ans,model_ans):
  return (0.85*semantic_score(model_ans,user_ans)+0.1*keyword_marks(model_ans,user_ans)+0.05*grammer_marks(user_ans))

def preprocess(text):
  text = text.lower()
  text = [s for s in text.split() if s.isalpha()]
  return " ".join(text)

@app.route('/')
def hello():
    return "hello world"  

@app.route("/predict", methods=['GET'])
def home():
  data = request.json['data']
  user_ans = data[0]["user_ans"]
  model_ans = data[1]["model_ans"]
  report_card = []
  marks = get_total(user_ans,model_ans)
  dictionary = {
            "grade":marks
        }
  report_card.append(dictionary)
  
  return jsonify({
      "report_card": report_card
  })


if __name__=='__main__':
    app.run()
