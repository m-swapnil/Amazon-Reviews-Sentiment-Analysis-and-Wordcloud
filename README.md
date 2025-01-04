# Amazon-Reviews-Sentiment-Analysis-and-Wordcloud


This repository contains a Python script that performs sentiment analysis and generates word clouds from Amazon product reviews. The script processes a sample dataset of reviews, cleans the text, performs sentiment analysis, and generates visualizations such as word clouds for positive, negative, and all reviews. The script also includes bigram analysis to visualize frequent two-word combinations.

## Requirements

### Python 3.x

### Libraries:
- `pandas` for data manipulation.
- `nltk` for natural language processing tasks.
- `wordcloud` for generating word clouds.
- `matplotlib` for plotting the word clouds and graphs.
- `re` for regular expressions to clean the text.
- `sklearn` for bigram analysis.

You can install the necessary libraries using the following:

pip install pandas nltk wordcloud matplotlib scikit-learn
How to Run the Code
Set Up:
Download the Amazon product review CSV file (e.g., amazon_com-product_reviews__20200101_20200331_sample.csv).

The dataset should contain a Review Title column or equivalent, which holds the review text.
NLTK Data:

The script requires specific datasets from NLTK (punkt, stopwords, and vader_lexicon). You can download them by running the following commands:
python
Copy code
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
Running the Script:
After downloading the dataset and installing dependencies, replace the CSV path in the script with the location of your dataset, and simply run the Python script:

python amazon_reviews.py
Expected Output:
Word Cloud Generation:

Word clouds will be generated and displayed for:
General Word Cloud: The most frequent words from all reviews.
Positive Word Cloud: Words associated with positive sentiment.
Negative Word Cloud: Words associated with negative sentiment.
Bigram Word Cloud: Most frequent two-word combinations in the reviews.
Sentiment Analysis:

The script also generates sentiment scores using VADER and applies sentiment analysis to the reviews, categorizing them into positive, neutral, and negative scores.
WordClouds and Sentiment Analysis
WordClouds:
The script generates multiple word clouds:

General WordCloud: Shows the most frequent words in the reviews.
Positive WordCloud: Shows words from positive sentiment reviews.
Negative WordCloud: Shows words from negative sentiment reviews.
Bigram WordCloud: Displays the most frequent two-word combinations (bigrams) in the reviews.
Text Cleaning and Preprocessing:
The script cleans and preprocesses the text in the following ways:

Remove unwanted characters: Non-alphabetic characters are removed.
Lowercase: Text is converted to lowercase for uniformity.
Remove stopwords: Common English stopwords (e.g., "the", "and") are removed.
Lemmatization: Words are lemmatized to their root forms (e.g., "running" becomes "run").
Bigram Analysis: Generates bigrams (two-word combinations) and visualizes their frequencies.
Output Files:
WordCloud Images: WordClouds for positive, negative, and all reviews will be displayed using matplotlib.
Review Dataset: The script does not save reviews to an external file, but you can modify the code to do so if needed.
Example Screenshots of WordClouds:
General WordCloud: A visualization of the most frequent words in all reviews.
Positive WordCloud: Shows words associated with positive sentiment.
Negative WordCloud: Shows words associated with negative sentiment.
Bigram WordCloud: Displays the most frequent two-word combinations.
Important Notes
Amazon Review Dataset: This script assumes you have a CSV file with Amazon product reviews. Ensure the dataset has a proper column for review titles or adjust the column name accordingly in the script.
Legal and Ethical Implications: If you are scraping data from Amazon, be aware of the legal and ethical implications. Ensure that scraping adheres to Amazon’s terms of service or consider using legitimate API access for fetching data.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Credits
This project uses the following libraries:

Pandas: For data manipulation and cleaning.
NLTK: For natural language processing tasks, including sentiment analysis.
WordCloud: For generating word clouds.
Matplotlib: For displaying word clouds and visualizations.
Scikit-learn: For bigram analysis and vectorization.
