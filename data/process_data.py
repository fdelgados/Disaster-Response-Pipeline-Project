import re
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories data and creates a merged dataframe

    Args:
        messages_filepath (str): Path to the messages file
        categories_filepath (str): Path to the categories file

    Returns:
        (pd.DataFrame): A messages and categories dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages.merge(categories, on='id')


def replace_url(text, replace=''):
    """Replaces all urls with the replacement string

    Args:
        text (str): The string being searched and replaced on
        replace (str): The replacement value that replaces found urls

    Returns:
        str: text with the replaced urls
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, replace)

    return text


def clean_data(df):
    """
    Performs the data clean process
        * Creates a column by category
        * Drops duplicated rows
        * Searches and replaces the url from the messages
        * Drops columns with one unique value
    Args:
        df (pd.DataFrame): Dataframe to be cleaned

    Returns:
        pd.DataFrame: clean dataframe
    """
    categories = df['categories'].str.split(';', expand=True)

    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    categories.columns = row.apply(lambda x: x[:-2]).tolist()

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        numeric_value = pd.to_numeric(categories[column].apply(lambda x: '1' if x == '1' else '0'))
        categories[column] = numeric_value

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1)

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()

    # remove url from messages
    df['message'] = df['message'].apply(lambda text: replace_url(text))

    for column in categories:
        # drop columns with one unique value
        if df[column].nunique() == 1:
            df.drop(column, axis=1, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save clean data to a database

    Args:
        df (pd.DataFrame): clean dataframe
        database_filename (str): Path to the database file
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))

    df.to_sql('messages', con=engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
