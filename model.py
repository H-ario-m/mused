from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
import numpy as np

# Load the model
model = load_model('emotion_detection_model.h5')

# Load the tokenizer
with open('emotion_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)
def preprocess_input(user_input):
    # Convert the text to sequences and pad it
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    return padded_sequence
def predict_emotion(user_input):
    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)
    print(f"Prediction: {prediction}")
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    print(f"Predicted Emotion: {predicted_label[0]}")
    return predicted_label[0]
def generate_response(emotion):
    responses = {
        'happiness': "I'm glad to hear that you're feeling happy! 😊",
        'sadness': "I'm here for you. It's okay to feel sad sometimes. 😔",
        'anger': "It sounds like something is frustrating you. I'm here to listen.",
        'surprise': "Wow, that sounds surprising! 😮",
        'fear': "It's okay to feel scared sometimes. I'm here with you. 😟",
        'love': "its great to have someone to love! ❤️",
        'thankfulness': "You're welcome! I'm happy to help. 😊",
        'joy': "Joy is such a beautiful feeling! 😄",
        'hate': "i'll be more consideratre from now on 🥺",
        'worry':"don't worry i am here for you ☺️"
        # Continue for all emotion labels in your dataset...
    }
    return responses.get(emotion, "I'm not sure how to respond to that emotion, but I'm here to help!")

while True:
    user_input = input("You: ")
    emotion = predict_emotion(user_input)
    response = generate_response(emotion)
    print("Bot:", response)

