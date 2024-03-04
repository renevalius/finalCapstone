# sentiment_analysis.py

import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Add SpacyTextBlob to the pipeline
nlp.add_pipe('spacytextblob')

def preprocess_text(text):
    """
    Preprocess text by removing stopwords and punctuation, and performing lowercasing.
    """
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given piece of text, returning the polarity and subjectivity.
    """
    doc = nlp(text)
    return doc._.blob.polarity, doc._.blob.subjectivity

def main():
    # Load the dataset
    data = pd.read_csv('amazon_product_reviews.csv')

    # Preprocess the reviews
    data['processed_reviews'] = data['reviews.text'].dropna().apply(preprocess_text)

    # Perform sentiment analysis
    data[['polarity', 'subjectivity']] = data['processed_reviews'].apply(
        lambda x: pd.Series(analyze_sentiment(x)))

    # Display the first few rows of the dataframe to verify
    print(data[['reviews.text', 'processed_reviews', 'polarity', 'subjectivity']].head())

    # Save the results to a new CSV file
    data.to_csv('processed_amazon_product_reviews.csv', index=False)
    print("Sentiment analysis completed. Results saved to 'processed_amazon_product_reviews.csv'.")

if __name__ == "__main__":
    main()
