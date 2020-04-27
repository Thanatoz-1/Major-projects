#Usage: python app.py
import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# import the necessary packages
import torch
from torch import nn
from torchtext import data
import torch.nn.functional as F

import urllib.request as urllib2
import pickle
import numpy as np
import argparse
import time
import uuid
import base64

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text):
        
        #text = [sent len, batch size]
        embedded = self.embedding(text)
                
        #embedded = [sent len, batch size, emb dim]
        embedded = embedded.permute(1, 0, 2)
        
        #embedded = [batch size, sent len, emb dim]
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        #pooled = [batch size, embedding_dim]
        return self.fc(pooled)

INPUT_DIM = 25002
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = 1


stoi = torch.load('models/TEXT_stoi.h5')
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
model.load_state_dict(torch.load('models/model_weights.h5'))
model.eval()

import spacy
try:
    nlp = spacy.load('en_core_web_sm')
except:
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


def predict(model, sentence):
    model.eval()
    tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(sentence)])
    indexed = [stoi.get(t,0) for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
    

print('PREDICTION FROM MODEL NEG: ', predict(model, 'What a lovely piece of shit this was'))
print('PREDICTION FROM MODEL POS: ', predict(model, 'This film is great'))

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='Sentiment Analysis', imagesource="https://hotemoji.com/images/dl/c/confused-face-emoji-by-twitter.png")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.form['u']
        print(file)
        
        result = predict(model, file)
    
        print(result)
        print("--- %s seconds ---" % str (time.time() - start_time))
        label = 'https://hotemoji.com/images/dl/9/grin-emoji-by-twitter.png' if result>0.5 else 'https://hotemoji.com/images/dl/x/sad-emoji-by-twitter.png'
        res='Happy' if result>0.5 else "Sad"
        return render_template('template.html', label=res, imagesource=label, sentence=file)

from flask import send_from_directory

if __name__ == "__main__":
    app.debug=False
app.run(host='0.0.0.0', port=5000, debug=True)