# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

app = Flask(__name__)

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./smart_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_smart_scores(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    score = probs.detach().numpy()[0][1]  # Get the probability of the positive class
    return score

#get_smart_scores("The purpose of this project is to address the growing need for efficient personal finance management tools and to capture a significant share of the mobile app market.")

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    text = data.get('text')
    criteria = {
        "Specific": get_smart_scores(f"Is this goal specific? {text}"),
        "Measurable": get_smart_scores(f"Is this goal measurable? {text}"),
        "Achievable": get_smart_scores(f"Is this goal achievable? {text}"),
        "Relevant": get_smart_scores(f"Is this goal relevant? {text}"),
        "Time-bound": get_smart_scores(f"Is this goal time-bound? {text}")
    }
    
    print("\nEvaluation of your project goal against SMART criteria:")
    for criterion, score in criteria.items():
        print(f"{criterion}: {score:.2f}")
    
    if all(score > 0.4 for score in criteria.values()):
        print("\nYour goal meets the SMART criteria!")
    else:
        print("\nYour goal does not fully meet the SMART criteria.")
    for criterion, score in criteria.items():
        if score <= 0.4:
            print(f"- The goal is not {criterion.lower()}.")
    

    return jsonify(criteria)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
