import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import csv



data_train = 'survey.csv'

sentences_train = []
labels_train = []
separator = ' '
with open (data_train, 'r', encoding='utf8') as csvfile:
  sentences = csv.reader(csvfile, delimiter=',')
  next(sentences)
  for row in sentences:
    sentences_train.append(separator.join(row[2:]))
    labels_train.append(str(row[1]))


le = preprocessing.LabelEncoder()
labels_train = le.fit_transform(labels_train)

vocab_size = 10000
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Generate the word index dictionary
tokenizer.fit_on_texts(sentences_train)
word_index = tokenizer.word_index

# print(labels_train)
def pad_input(input):
    sequence = tokenizer.texts_to_sequences(input)
    padded = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    return padded

def predict(input):
    model = tf.keras.models.load_model('saved_model/1')
    result = model.predict([pad_input(input)])

    return le.inverse_transform([np.argmax(result)])