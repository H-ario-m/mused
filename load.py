from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the saved model
model = load_model('emotion_detection_model.h5')

# Load the tokenizer
with open('emotion_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Example input for prediction
new_tweet = ["why do boys wrong things to girls!it makes ,me angry"]
new_tweet_seq = tokenizer.texts_to_sequences(new_tweet)
new_tweet_padded = pad_sequences(new_tweet_seq, maxlen=100, padding='post')

# Predict the emotion
predicted_class = model.predict(new_tweet_padded)
predicted_label = label_encoder.inverse_transform([predicted_class.argmax(axis=1)[0]])

print(f"Predicted emotion: {predicted_label[0]}")
