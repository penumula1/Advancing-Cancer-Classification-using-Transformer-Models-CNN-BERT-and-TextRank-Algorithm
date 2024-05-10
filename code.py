import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Path to the CSV file
csv_file = r"E:\TCGA_Reports.csv"

# Initialize NLTK
nltk.download('punkt')

# Function to calculate cosine similarity
def calculate_cosine_similarity(reference_summary, text):
    sentences = sent_tokenize(text)
    vectorizer = CountVectorizer().fit_transform([reference_summary] + sentences)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return sentences[cosine_similarities.argmax()]

# Read the CSV file
df = pd.read_csv(csv_file)

# Reference summary (if available)
reference_summary = "This is a reference summary."

# Initialize a list to store summaries
summaries = []

# Iterate over rows and calculate summaries
for idx, row in df.iterrows():
    text = row['text']
    summary = calculate_cosine_similarity(reference_summary, text)
    summaries.append(summary)

# Display the summaries
for idx, summary in enumerate(summaries, start=1):
    print(f"Summary {idx}: {summary}")


import matplotlib.pyplot as plt

# Calculate cosine similarity scores
cosine_similarity_scores = []
for text in df['text']:
    summary = calculate_cosine_similarity(reference_summary, text)
    vectorizer = CountVectorizer().fit_transform([reference_summary, summary])
    vectors = vectorizer.toarray()
    cosine_similarity_scores.append(cosine_similarity(vectors)[0,1])

# Plot the cosine similarity scores
plt.figure(figsize=(10, 6))
plt.bar(range(len(cosine_similarity_scores)), cosine_similarity_scores)
plt.xlabel('Summary Index')
plt.ylabel('Cosine Similarity Score')
plt.title('Cosine Similarity Scores of Generated Summaries')
plt.show()

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

# Path to the CSV file
csv_file = r"E:\TCGA_Reports.csv"

# Initialize NLTK
nltk.download('punkt')

# Function to calculate cosine similarity
def calculate_cosine_similarity(reference_summary, text):
    sentences = sent_tokenize(text)
    vectorizer = CountVectorizer().fit_transform([reference_summary] + sentences)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return sentences[cosine_similarities.argmax()]

# Function to calculate BLEU score
def calculate_bleu_score(reference_summary, generated_summary):
    reference = reference_summary.split()
    generated = generated_summary.split()
    return sentence_bleu([reference], generated)

# Read the CSV file
df = pd.read_csv(csv_file)

# Reference summary (if available)
reference_summary = "This is a reference summary."

# Initialize lists to store summaries
summaries = []

# Initialize Rouge
rouge = Rouge()

# Initialize lists to store scores
cosine_similarity_scores = []
rouge_scores = []
bleu_scores = []

# Iterate over rows and calculate summaries and scores
for idx, row in df.iterrows():
    text = row['text']
    summary = calculate_cosine_similarity(reference_summary, text)
    summaries.append(summary)
    
    # Calculate cosine similarity score
    cosine_similarity_scores.append(calculate_cosine_similarity(reference_summary, text))
    
    # Calculate ROUGE score
    rouge_scores.append(rouge.get_scores(summary, reference_summary)[0]['rouge-1']['f'])
    
    # Calculate BLEU score
    bleu_scores.append(calculate_bleu_score(reference_summary, summary))

# Display the summaries and scores
for idx, summary in enumerate(summaries, start=1):
    print(f"Summary {idx}: {summary}")
    print(f"Cosine Similarity Score: {cosine_similarity_scores[idx - 1]}")
    print(f"ROUGE Score: {rouge_scores[idx - 1]}")
    print(f"BLEU Score: {bleu_scores[idx - 1]}")
    print()

# Calculate average scores
avg_cosine_similarity = sum(cosine_similarity_scores) / len(cosine_similarity_scores)
avg_rouge = sum(rouge_scores) / len(rouge_scores)
avg_bleu = sum(bleu_scores) / len(bleu_scores)

print(f"Average Cosine Similarity Score: {avg_cosine_similarity}")
print(f"Average ROUGE Score: {avg_rouge}")
print(f"Average BLEU Score: {avg_bleu}")


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
df = pd.read_csv(r"E:\TCGA_Reports.csv")

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])

X = tokenizer.texts_to_sequences(df['text'])

# Determine the max length of sequences
max_len = max(len(seq) for seq in X)

# Pad sequences to ensure uniform length
X = pad_sequences(X, maxlen=max_len, padding='post')

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train-test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Define labels (assuming binary classification)
# Replace this with your actual label data
y_train = np.random.randint(2, size=(X_train.shape[0],))
y_test = np.random.randint(2, size=(X_test.shape[0],))

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')


import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Generate sample data
np.random.seed(0)
data = np.random.normal(loc=100, scale=20, size=1000)  # Generate 1000 random numbers with mean=100 and std=20

# Calculate mean, median, and mode
mean = np.mean(data)
median = np.median(data)
mode = float(stats.mode(data)[0])  # Mode returns an array, so we take the first element

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black', density=True)
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
plt.axvline(mode, color='orange', linestyle='dashed', linewidth=2, label=f'Mode: {mode:.2f}')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Histogram of Sample Data with Mean, Median, and Mode')
plt.legend()
plt.grid(True)
plt.show()


import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv(r"E:\TCGA_Reports.csv")

# Tokenize the text into sentences
df['sentences'] = df['text'].apply(sent_tokenize)

# Tokenize the text into words
df['words'] = df['text'].apply(word_tokenize)

# Create a Bag of Words representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Calculate pairwise cosine similarity
cosine_sim = cosine_similarity(X, X)

# Implement TextRank algorithm for text summarization
def textrank_summary(text, num_sentences=2):
    sentences = sent_tokenize(text)
    word_embeddings = vectorizer.transform(sentences)
    sentence_scores = cosine_similarity(word_embeddings, X).sum(axis=1)
    ranked_sentences = [sentence for _, sentence in sorted(zip(sentence_scores, sentences), reverse=True)[:num_sentences]]
    return ' '.join(ranked_sentences)

# Generate summaries for each text
df['summary'] = df['text'].apply(lambda x: textrank_summary(x))

# Print the summaries
for idx, summary in enumerate(df['summary']):
    print(f"Summary for text {idx+1}:")
    print(summary)
    print()
