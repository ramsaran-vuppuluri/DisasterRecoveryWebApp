import json
import plotly
import pandas as pd

import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterRecovery.db')
df = pd.read_sql_table('DisasterRecovery', engine)

df.related.replace(to_replace=2, value=1, inplace=True)

# load model
model = joblib.load("../models/Classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = df.columns.drop(['id', 'message', 'original', 'genre'])
    category_counts = df[category_names].sum()

    direct_category_counts = df[df.genre == 'direct'][category_names].sum()
    news_category_counts = df[df.genre == 'news'][category_names].sum()
    social_category_counts = df[df.genre == 'social'][category_names].sum()

    category_names = [re.sub('[^a-zA-Z0-0]', ' ', category).capitalize().strip() for category in category_names]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )

            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_counts,
                    y=category_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'autosize': True,
                'height': 1000,
                'xaxis': {
                    'title': "Count",
                    'automargin': True
                },
                'yaxis': {
                    'automargin': True
                }
            }
        },
        {
            'data': [
                Bar(
                    x=direct_category_counts,
                    y=category_names,
                    orientation='h',
                    name='Direct'
                ),
                Bar(
                    x=news_category_counts,
                    y=category_names,
                    orientation='h',
                    name='News'
                ),
                Bar(
                    x=social_category_counts,
                    y=category_names,
                    orientation='h',
                    name='Social'
                )
            ],
            'layout': {
                'title': 'Distribution of Categories by Genre',
                'autosize': True,
                'height': 1000,
                'barmode': 'stack',
                'xaxis': {
                    'title': "Count",
                    'automargin': True
                },
                'yaxis': {
                    'automargin': True
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
