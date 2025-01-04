# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 20:51:15 2025

@author: Swapnil Mishra
"""

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK data if you haven't already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load CSV file
df = pd.read_csv("C:/Users/Swapnil Mishra/Desktop/DS/Text Mining/Amazon review scrapper and wordcloud/amazon_com-product_reviews__20200101_20200331_sample.csv")  # Replace with your actual CSV path

# Display the first few rows of the dataframe
print(df.head())

# Extract the reviews column (assuming 'Comments' column holds the review text)
reviews = df['Review Title'].astype(str).tolist()

# Define a function to clean the text
def clean_text(text):
    # Remove unwanted characters and symbols
    text = re.sub(r"[^A-Za-z\s]", "", text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Clean the reviews
cleaned_reviews = [clean_text(review) for review in reviews]

# Join all the cleaned reviews into one string
text = " ".join(cleaned_reviews)

# Generate the word cloud for all reviews
wordcloud = WordCloud(
    background_color="white",
    width=1500,
    height=1000,
    max_words=100,
).generate(text)

# Plot the word cloud for all reviews
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')  # Turn off axis
plt.title("Word Cloud for All Reviews")
plt.show()

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Apply sentiment analysis to each review
df['sentiments'] = df['Review Title'].apply(lambda x: sid.polarity_scores(x))

# Convert sentiment to separate columns for positive, neutral, and negative scores
df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

# Display the dataframe with sentiment scores
print(df.head())

# Positive reviews (e.g., sentiment > 0.05)
positive_reviews = df[df['pos'] > 0.05]['Review Title'].astype(str).tolist()
positive_text = " ".join([clean_text(review) for review in positive_reviews])

# Generate positive word cloud
positive_wordcloud = WordCloud(background_color='white', width=1500, height=1000).generate(positive_text)
plt.figure(figsize=(10, 8))
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Positive Word Cloud")
plt.show()

# Negative reviews (e.g., sentiment < -0.05)
negative_reviews = df[df['neg'] > 0.05]['Review Title'].astype(str).tolist()
negative_text = " ".join([clean_text(review) for review in negative_reviews])

# Generate negative word cloud
negative_wordcloud = WordCloud(background_color='black', width=1500, height=1000).generate(negative_text)
plt.figure(figsize=(10, 8))
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Negative Word Cloud")
plt.show()

# Additional Analysis: Bigram WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Lowercase and tokenize the text
text = text.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars as well as stop words.
stopwords_wc = set(stopwords.words('english'))
customised_words = ['iphone', 'apple', 'product']  # Add any specific stop words

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text1 if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Lemmatize words
WNL = nltk.WordNetLemmatizer()
text_content = [WNL.lemmatize(t) for t in text_content]

# Create bigrams from the tokenized text
bigrams_list = list(nltk.bigrams(text_content))

# Using CountVectorizer to view the frequency of bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform([' '.join(bigram) for bigram in bigrams_list])

# Sum the word frequencies
sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Generate wordcloud for bigrams
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)

wordCloud.generate_from_frequencies(words_dict)
plt.figure(figsize=(12, 8))
plt.title('Most Frequent Bigrams Word Cloud')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()
