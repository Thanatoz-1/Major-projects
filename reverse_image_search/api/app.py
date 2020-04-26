#Usage: python app.py
import os

from flask import Flask, render_template, request, redirect, url_for

# import the necessary packages
import torch
from torch import nn
import torchvision 
from torchvision import models, transforms
from PIL import Image

import urllib.request as urllib2
import pickle
import numpy as np
import argparse
import time
import uuid
import base64


image_size = 224
print(Image.open('uploads/template.jpg'))
device = 'cpu'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = ResNet101(weights=model_weights_path)
mappings = pickle.load(
    urllib2.urlopen(
        'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'
        ) 
    )
model = models.resnet50(pretrained=True).to(device)
model.eval()


transform = transforms.Compose([            
     transforms.Resize(image_size),             
     transforms.ToTensor(),                     
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(request.get(url).content)

def predict(file):
    x = image=Image.open(file)
    img_t = transform(image)
    image = torch.unsqueeze(img_t, 0)
    # Carry out model inference
    out = model(image.to(device))
    # Forth, print the top 5 classes predicted by the model
    _, indices = torch.sort(out, descending=True)
    
    return mappings[indices[0][0].item()]

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='Reverse Image Search', imagesource='uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            result = predict(file_path)
        
            print(result)
            print(file_path)
            filename = my_random_string(6) + file.filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', label=result, imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug.middleware.shared_data import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)