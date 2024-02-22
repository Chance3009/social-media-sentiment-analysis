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
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    country_distribution = {}

    hashtag = request.form['hashtag']

    df = pd.read_csv('social_media_data.csv', encoding='latin1')

    # Filter post with specific hashtag input by user
    filtered_df = df[df['Hashtags'].str.contains(hashtag, case=False)]

    data = filtered_df['Text'].tolist()
    countries = filtered_df['Country'].tolist()
    for i in range(len(data)):
        result = sentiment_classifier(data[i])[0]['label']
        if result == 'positive':
            sentiment_counts['positive'] += 1
        elif result == 'negative':
            sentiment_counts['negative'] += 1
        else:
            sentiment_counts['neutral'] += 1

        # Update country_distribution
        country = countries[i]
        if country not in country_distribution:
            country_distribution[country] = {
                'positive': 0, 'negative': 0, 'neutral': 0}

        # Check if the sentiment is present in the dictionary before updating
        if result in country_distribution[country]:
            country_distribution[country][result] += 1

        # Calculate the average number of likes and retweets
        average_likes = filtered_df['Likes'].mean()
        average_retweets = filtered_df['Retweets'].mean()

    return render_template('dashboard.html', hashtag=hashtag, sentiment_counts=sentiment_counts, country_distribution=country_distribution, countries=countries, average_likes=average_likes, average_retweets=average_retweets)


if __name__ == '__main__':
    app.run(debug=True)
