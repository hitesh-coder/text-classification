import os
from flask import Flask, render_template
from flask import request

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "ivanlau/distil-bert-uncased-finetuned-github-issues"

tokenizer = AutoTokenizer.from_pretrained(name)

model = AutoModelForSequenceClassification.from_pretrained(name)

def answer_classification(paragraph):

    inputs = tokenizer(paragraph, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
      form = request.form
      result = []
      bert_abstract = form['paragraph']
      result.append(answer_classification(bert_abstract))

    #   return render_template("/content/index.html",result = result)
    #   question = form['question']
    #   result.append(form['question'])
    #   result.append(answer_question(question, bert_abstract))
    #   result.append(form['paragraph'])

      return render_template("index.html",result = result)

    # answer = answer_question(question, bert_abstract)

    # return "answer"
    return render_template("index.html")

    # return 'Hello, World!'


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)