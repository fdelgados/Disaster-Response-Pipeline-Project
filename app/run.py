import json
import plotly
import pandas as pd

import nltk

nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

stop_words = set(stopwords.words('english'))


def tokenize(text):
    """Convert a text to a token list in lowercase and lemmatized. It also eliminates stop words.

    Args:
        text (str): Text to be converted

    Returns:
        list: A list of clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(token.strip()).lower() for token in tokens
            if token.lower() not in stop_words]


def create_graphs(dataframe):
    """Extracts and format data to create visualizations

    Args:
        dataframe (pd.DataFrame): Visualizations data source

    Returns:
        (list, str)
    """
    # extract data needed for visuals
    genre_counts = dataframe.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = dataframe.iloc[:, 4:].sum()
    category_names = [category.replace('_', ' ').title() for category in dataframe.iloc[:, 4:].columns.tolist()]

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
                'title': 'Distribution of Message Categories',
                'xaxis': {
                    'title': 'Count'
                },
                'yaxis': {
                    'title': 'Category',
                    'automargin': True
                },
            }
        }
    ]
    # create visuals

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return ids, graph_json


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    ids, graph_json = create_graphs(df)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graph_json=graph_json)


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
    app.run(host='localhost', port=3001, debug=True)


if __name__ == '__main__':
    main()
