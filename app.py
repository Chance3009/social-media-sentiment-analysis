from flask import Flask, render_template, request
import pandas as pd
from transformers import pipeline

app = Flask(__name__)

sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    hashtag = request.form['hashtag']

    df = pd.read_csv('social_media_data.csv', encoding='latin1')

    filtered_df = df[df['Hashtags'].str.contains(hashtag, case=False)]

    data = filtered_df['Text'].tolist()

    for i in range(len(data)):
        result = sentiment_classifier(data[i])[0]['label']
        if result == 'positive':
            sentiment_counts['Positive'] += 1
        elif result == 'negative':
            sentiment_counts['Negative'] += 1
        else:
            sentiment_counts['Neutral'] += 1

    return render_template('dashboard.html', hashtag=hashtag, sentiment_counts=sentiment_counts)


if __name__ == '__main__':
    app.run(debug=True)
