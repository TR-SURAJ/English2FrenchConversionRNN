# -*- coding: utf-8 -*-
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import load_model

from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='simple_rnn_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

#Load the french tokenizer
with open('fr_tokenizer.pickle', 'rb') as handle:
    fr_tokenizer = pickle.load(handle)
    
#Load the french tokenizer
with open('en_tokenizer.pickle', 'rb') as handle:
    en_tokenizer = pickle.load(handle)


def tokenize(x):
   tokenizer = en_tokenizer
   tokenizer.fit_on_texts(x)
   return tokenizer.texts_to_sequences(x)

def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    if(request.method == 'POST'):
        message = request.form['message']
        print('the message',message)
        tokenized_en = tokenize([message])
        print('the tokenized message',tokenized_en)
        padded_en = pad(tokenized_en, 21)
        print('padded',padded_en)
        padded_en = padded_en.reshape((-1, 21))
        print('2 dim pad',padded_en)
        my_prediction = logits_to_text(model.predict(padded_en)[0], fr_tokenizer)
        print('The prediction',my_prediction)
        
    return render_template('result.html',prediction = my_prediction)
        

if __name__ == '__main__':
	app.run(debug=True)
        


