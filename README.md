# Amazon Product Review Analysis and WordCloud Generation

This repository contains a Python script that processes Amazon product reviews, performs sentiment analysis, and generates visualizations using WordClouds. The script analyzes reviews based on sentiment, categorizing them into positive and negative reviews, and generates word clouds for each category. It also provides a general word cloud for all reviews.

## Features
- Clean reviews to remove unwanted characters.
- Perform sentiment analysis using VADER.
- Generate WordClouds for all reviews, positive reviews, negative reviews, and bigrams.
- Visualize common words and bigrams using WordClouds.

---

## Prerequisites

Before using the script, ensure you have the following:

1. Python 3.6 or above installed on your machine.
2. Required Python libraries:
   ```bash
   pip install pandas nltk wordcloud matplotlib scikit-learn
   ```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/amazon-review-analysis.git
   cd amazon-review-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Ensure that your CSV file containing Amazon reviews is in the correct format. The script assumes the reviews are in a column named `Review Title`. Modify the script if your CSV structure differs.

2. Run the script:
   ```bash
   python amazon_review_analysis.py
   ```

---

## Example Output

- WordClouds for:
  - All reviews.
  - Positive reviews.
  - Negative reviews.
  - Bigram frequency (most frequent two-word combinations).

- Dataframe with sentiment scores for each review (positive, neutral, and negative).

---

## File Structure
```
.
|-- amazon_review_analysis.py   # Main script to process reviews and generate word clouds
|-- requirements.txt            # Required Python packages
|-- reviews.csv                 # Sample CSV with Amazon reviews (generated at runtime)
```

---

## License
This project is licensed under the MIT License.
