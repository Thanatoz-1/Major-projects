# Sentiment Analysis

In this notebook we are going to achieve a decent test accuracy of ~84% using all of the common techniques used for sentiment analysis. In this notebook, we'll implement a model that gets comparable results whilst training significantly faster and using around half of the parameters. More specifically, we'll be implementing the "FastText" model from the paper [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759).

## Preparing Data

One of the key concepts in the FastText paper is that they calculate the n-grams of an input sentence and append them to the end of a sentence. Here, we'll use bi-grams. Briefly, a bi-gram is a pair of words/tokens that appear consecutively within a sentence. 

For example, in the sentence "how are you ?", the bi-grams are: "how are", "are you" and "you ?".

The `generate_bigrams` function takes a sentence that has already been tokenized, calculates the bi-grams and appends them to the end of the tokenized list.


```python
def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x
```

As an example:


```python
generate_bigrams(['This', 'film', 'is', 'terrible'])
```

TorchText `Field`s have a `preprocessing` argument. A function passed here will be applied to a sentence after it has been tokenized (transformed from a string into a list of tokens), but before it has been numericalized (transformed from a list of tokens to a list of indexes). This is where we'll pass our `generate_bigrams` function.

As we aren't using an RNN we can't use packed padded sequences, thus we do not need to set `include_lengths = True`.


```python
!pip install torchtext
!pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
```

    Collecting https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
      Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz (12.0 MB)
    [K     |████████████████████████████████| 12.0 MB 1.8 MB/s eta 0:00:01
    [?25hRequirement already satisfied (use --upgrade to upgrade): en-core-web-sm==2.2.0 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages
    Requirement already satisfied: spacy>=2.2.0 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from en-core-web-sm==2.2.0) (2.2.3)
    Requirement already satisfied: numpy>=1.15.0 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.18.1)
    Requirement already satisfied: thinc<7.4.0,>=7.3.0 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (7.3.1)
    Requirement already satisfied: srsly<1.1.0,>=0.1.0 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.0.1)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.1.3)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (3.0.2)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (2.0.3)
    Requirement already satisfied: setuptools in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (45.2.0)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.0.2)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (1.0.0)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (2.23.0)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (0.6.0)
    Requirement already satisfied: blis<0.5.0,>=0.4.0 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from spacy>=2.2.0->en-core-web-sm==2.2.0) (0.4.1)
    Requirement already satisfied: tqdm<5.0.0,>=4.10.0 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from thinc<7.4.0,>=7.3.0->spacy>=2.2.0->en-core-web-sm==2.2.0) (4.43.0)
    Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.0->en-core-web-sm==2.2.0) (1.5.0)
    Requirement already satisfied: chardet<4,>=3.0.2 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->en-core-web-sm==2.2.0) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->en-core-web-sm==2.2.0) (2019.11.28)
    Requirement already satisfied: idna<3,>=2.5 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->en-core-web-sm==2.2.0) (2.9)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.0->en-core-web-sm==2.2.0) (1.24.3)
    Requirement already satisfied: zipp>=0.5 in /home/thanoz/.virtualenvs/kaggle/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.0->en-core-web-sm==2.2.0) (3.0.0)
    Building wheels for collected packages: en-core-web-sm
      Building wheel for en-core-web-sm (setup.py) ... [?25ldone
    [?25h  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.0-py3-none-any.whl size=12019121 sha256=49408f67e82672c1d5df55e09aaad2f675415fc35c071429efac6b0b0b867889
      Stored in directory: /home/thanoz/.cache/pip/wheels/1d/bc/94/171b09b7fcce517723f40606754e5b7374770cc39290e092bf
    Successfully built en-core-web-sm



```python
import torch
from torchtext import data
from torchtext import datasets

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', preprocessing = generate_bigrams, tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype = torch.float)
```

As before, we load the IMDb dataset and create the splits.


```python
import random

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
```

Build the vocab and load the pre-trained word embeddings.


```python
MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.6B.100d", 
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)
```

And create the iterators.


```python
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)
```

## Build the Model

This model has far fewer parameters than the previous model as it only has 2 layers that have any parameters, the embedding layer and the linear layer. There is no RNN component in sight!

Instead, it first calculates the word embedding for each word using the `Embedding` layer (blue), then calculates the average of all of the word embeddings (pink) and feeds this through the `Linear` layer (silver), and that's it!

![](assets/sentiment8.png)

We implement the averaging with the `avg_pool2d` (average pool 2-dimensions) function. Initially, you may think using a 2-dimensional pooling seems strange, surely our sentences are 1-dimensional, not 2-dimensional? However, you can think of the word embeddings as a 2-dimensional grid, where the words are along one axis and the dimensions of the word embeddings are along the other. The image below is an example sentence after being converted into 5-dimensional word embeddings, with the words along the vertical axis and the embeddings along the horizontal axis. Each element in this [4x5] tensor is represented by a green block.

![](assets/sentiment9.png)

The `avg_pool2d` uses a filter of size `embedded.shape[1]` (i.e. the length of the sentence) by 1. This is shown in pink in the image below.

![](assets/sentiment10.png)

We calculate the average value of all elements covered by the filter, then the filter then slides to the right, calculating the average over the next column of embedding values for each word in the sentence. 

![](assets/sentiment11.png)

Each filter position gives us a single value, the average of all covered elements. After the filter has covered all embedding dimensions we get a [1x5] tensor. This tensor is then passed through the linear layer to produce our prediction.


```python
import torch.nn as nn
import torch.nn.functional as F

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
```

As previously, we'll create an instance of our `FastText` class.


```python
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
```

Looking at the number of parameters in our model, we see we have about the same as the standard RNN from the first notebook and half the parameters of the previous model.


```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')
```

    The model has 2,500,301 trainable parameters


And copy the pre-trained vectors to our embedding layer.


```python
pretrained_embeddings = TEXT.vocab.vectors

model.embedding.weight.data.copy_(pretrained_embeddings)
```




    tensor([[-0.1117, -0.4966,  0.1631,  ...,  1.2647, -0.2753, -0.1325],
            [-0.8555, -0.7208,  1.3755,  ...,  0.0825, -1.1314,  0.3997],
            [-0.0382, -0.2449,  0.7281,  ..., -0.1459,  0.8278,  0.2706],
            ...,
            [ 0.9262, -0.5523, -1.7168,  ..., -0.8267,  0.9865,  0.3870],
            [ 0.9085, -0.9059, -0.2949,  ...,  0.2342, -0.2620,  0.2176],
            [-0.0716,  0.1341,  0.2236,  ..., -0.1987,  0.4728,  0.6021]])



Not forgetting to zero the initial weights of our unknown and padding tokens.


```python
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
```

## Train the Model

Training the model is the exact same as last time.

We initialize our optimizer...


```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters())
```

We define the criterion and place the model and criterion on the GPU (if available)...


```python
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)
```

We implement the function to calculate accuracy...


```python
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc
```

We define a function for training our model...

**Note**: we are no longer using dropout so we do not need to use `model.train()`, but as mentioned in the 1st notebook, it is good practice to use it.


```python
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label)
        
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

We define a function for testing our model...

**Note**: again, we leave `model.eval()` even though we do not use dropout.


```python
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
```

As before, we'll implement a useful function to tell us how long an epoch takes.


```python
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```

Finally, we train our model.


```python
N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
```

    Epoch: 01 | Epoch Time: 0m 8s
    	Train Loss: 0.687 | Train Acc: 57.92%
    	 Val. Loss: 0.635 |  Val. Acc: 72.20%
    Epoch: 02 | Epoch Time: 0m 7s
    	Train Loss: 0.650 | Train Acc: 72.50%
    	 Val. Loss: 0.505 |  Val. Acc: 76.85%
    Epoch: 03 | Epoch Time: 0m 7s
    	Train Loss: 0.576 | Train Acc: 79.17%
    	 Val. Loss: 0.423 |  Val. Acc: 80.84%
    Epoch: 04 | Epoch Time: 0m 7s
    	Train Loss: 0.500 | Train Acc: 84.07%
    	 Val. Loss: 0.384 |  Val. Acc: 84.15%
    Epoch: 05 | Epoch Time: 0m 6s
    	Train Loss: 0.432 | Train Acc: 86.90%
    	 Val. Loss: 0.375 |  Val. Acc: 85.83%


...and get the test accuracy!

The results are comparable to the results in the last notebook, but training takes considerably less time!


```python
model.load_state_dict(torch.load('tut3-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
```

    Test Loss: 0.382 | Test Acc: 85.43%


## User Input

And as before, we can test on any input the user provides making sure to generate bigrams from our tokenized sentence.


```python
import spacy
nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence):
    model.eval()
    tokenized = generate_bigrams([tok.text for tok in nlp.tokenizer(sentence)])
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
```

An example negative review...


```python
predict_sentiment(model, "What a lovely piece of shit this was")
```




    0.00022014240676071495



An example positive review...


```python
predict_sentiment(model, "This film is great")
```




    1.0


