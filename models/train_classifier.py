import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

from warnings import simplefilter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

stop_words = set(stopwords.words('english'))


def load_data(database_filepath):
    """Loads data from the database.

    Args:
        database_filepath (str): Path to the database file

    Returns:
        tuple: A tuple containing messages as a first element, categories as a second element
            and a list of category names as the third element
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql_table('messages', con=engine)

    messages = df['message']
    categories = df.iloc[:, 4:]

    return messages, categories, categories.columns


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


def build_model():
    """Creates a model with optimized hyperparameters using MultinomialNB estimator

    Returns:
        A train ready model
    """
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__estimator__alpha': (1e-2, 1e-3)
    }

    return GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model and displays the accuracy, precision and the F1-score for each category

    Args:
        model: Model to be evaluated
        X_test (array): Messages from the test set
        Y_test (array): Labels from the test set
        category_names (list|pd.Series): List of category names
    """
    y_pred = model.predict(X_test)

    df_pred = pd.DataFrame(y_pred, columns=category_names)

    for category in category_names:
        print('Category {}:'.format(category))
        print(classification_report(Y_test[category], df_pred[category]))
        print('=' * 55)


def save_model(model, model_filepath):
    """Save the trained model to a file

    Args:
        model: Trained model
        model_filepath (str): Path to the model file
    """
    joblib.dump(model, model_filepath, compress=3)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
