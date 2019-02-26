'''
    Author: Ram Saran Vuppuluri.

        This Python script will act as ETL pipeline to:
            1. extract data from disaster_messages.csv and disaster_categories.csv files
            2. transform into one single dataset
            3. load into database as new table (will override if data is already present).

    Library versions:
        sqlalchemy:1.2.1
        Pandas:0.22.0

    Command to run:
        python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterRecovery.db
'''

import sys

import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on='id')

    categories = categories.categories.str.split(';', expand=True)

    row = categories[:1].values

    category_colnames = [item.split('-')[0] for value in row for item in value]

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    #df.drop(columns=['categories'], inplace=True)

    df.drop('categories',axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1, join='inner')

    return df


def clean_data(df):
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterRecovery', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
