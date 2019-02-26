'''
    Author: Ram Saran Vuppuluri.

        This Python script will act as ETL pipeline to:
            1. extract data from disaster_messages.csv and disaster_categories.csv files
            2. transform into one single dataset
            3. load into database as new table (will override if data is already present).

    Library versions:
        sqlalchemy: 1.2.1
        Pandas: 0.22.0

    Command to run:
        python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterRecovery.db
'''

import sys

import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This method will:
        1. Read messages and categories csv file and load them as Pandas DataFrame.
        2. Merge messages and categories DataFrames into one df DataFrame.
        3. Generate set of categories from the values in categories DataFrame.
        4. Clean and convert categories columns to numeric.
        5. Replace clean categories in df DataFrame.

    :param messages_filepath:
        Relative Filepath with file name of messages csv file.

    :param categories_filepath:
        Relative Filepath with file name of categories csv file.

    :return:
        Consolidated DataFrame.
    '''
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

    # df.drop(columns=['categories'], inplace=True)

    df.drop('categories', axis=1, inplace=True)

    df = pd.concat([df, categories], axis=1, join='inner')

    return df


def clean_data(df):
    '''
        This method will drop duplicate rows from the data frame.

    :param df:
        DataFrame

    :return:
        DataFrame with no duplicate rows.
    '''
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    This method will save DataFrame to SQL DB.

    :param df:
        DataFrame

    :param database_filename:
        Relative filepath where DB instance need to be persisted.

    :return:
        None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterRecovery', engine, index=False, if_exists='replace')


def main():
    '''
    This is the invocation method for this Python script.

    :return:
        None
    '''
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
