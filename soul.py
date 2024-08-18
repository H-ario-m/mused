import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import pickle

# Step 1: Load the dataset
tweet_emotions_path = r'M:\Downloads\DATASET\csv\tweet_emotions.csv' 
tweet_emotions_data = pd.read_csv(tweet_emotions_path)

# Step 2: Preprocess the text
tweet_emotions_data['content'] = tweet_emotions_data['content'].str.lower().str.replace('[^a-zA-Z0-9\s]', '')

# Tokenize the text
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(tweet_emotions_data['content'].values)

# Convert the text to sequences and pad them
sequences = tokenizer.texts_to_sequences(tweet_emotions_data['content'].values)
max_length = 100
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Step 3: Encode the sentiment labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tweet_emotions_data['sentiment'].values)

# Step 4: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Step 5: Build the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(128)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6: Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    batch_size=32,
    verbose=1
)

# Step 7: Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {accuracy}')

# Step 8: Save the model, tokenizer, and label encoder
model.save('emotion_detection_model.h5')

with open('emotion_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model, tokenizer, and label encoder have been saved successfully.")
# Load the saved model
model = load_model('emotion_detection_model.h5')

# Load the tokenizer
with open('emotion_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Example input for prediction
new_tweet = ["I am so happy today!"]
new_tweet_seq = tokenizer.texts_to_sequences(new_tweet)
new_tweet_padded = pad_sequences(new_tweet_seq, maxlen=100, padding='post')

# Predict the emotion
predicted_class = model.predict(new_tweet_padded)
predicted_label = label_encoder.inverse_transform([predicted_class.argmax(axis=1)[0]])

print(f"Predicted emotion: {predicted_label[0]}")
