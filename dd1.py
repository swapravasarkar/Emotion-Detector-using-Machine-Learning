import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon and initialize the sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def detect_emotion(text):
    # Get sentiment scores for the input text
    sentiment_scores = sia.polarity_scores(text)

    # Determine the emotion based on the sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

if __name__ == "__main__":
    # Input text for emotion detection
    input_text = input("Enter a sentence or text: ")

    # Detect and print the emotion
    emotion = detect_emotion(input_text)
    print(f"The detected emotion is: {emotion}")
