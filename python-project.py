import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os

# File paths for saving and loading the model
model_file_path = 'next_word_predictor_model2.h5'

# Function to load data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

# Function to preprocess data
def preprocess_data(data): 
    corpus = data.lower().split("\n")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = to_categorical(y, num_classes=total_words)

    return X, y, tokenizer, total_words, max_sequence_len

# Function to create model
def create_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    for _ in range(1):  # Generate 50 words
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=-1)[0]
        output_word = tokenizer.index_word[predicted_word_index]
        text += " " + output_word
    return text

# Load data
data = load_data('data.txt')

# Preprocess data
X, y, tokenizer, total_words, max_sequence_len = preprocess_data(data)

# Check if the model file exists
if os.path.exists(model_file_path):
    # Load the model if it exists
    model = tf.keras.models.load_model(model_file_path)
    print("Model loaded from disk.")
else:
    # Create and train the model if it doesn't exist
    model = create_model(total_words, max_sequence_len)
    history = model.fit(X, y, epochs=500, verbose=1)
    # Save the model after training
    model.save(model_file_path)
    print("Model trained and saved to disk.")

# Test the model with a seed text
seed_text = "people don't know what "  # Replace this with a relevant seed text
generated_text = predict_next_word(model, tokenizer, seed_text, max_sequence_len)
print("Generated text:", generated_text)
