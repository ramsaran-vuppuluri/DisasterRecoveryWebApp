'''
    Author: Ram Saran Vuppuluri.

        This Python script will act as Machine Learning training script. Following steps are performed in this script:
        1. Read from SQL table (this table is populated by process_data.py script
        2. Perform NLP ML pipeline.
        3. Evaluate model output with

    Library versions:
        sqlalchemy: 1.2.1
        Pandas: 0.22.0
        nltk: 3.2.5
        sklearn: 0.19.1

    Command to run:
        python3 train_classifier.py ../data/DisasterRecovery.db AdaBoostClassifier
'''

import datetime
import re
import sys
import time

import nltk
import pandas as pd
from sqlalchemy import create_engine

nltk.download('punkt')

from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline, make_union, make_pipeline

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report, accuracy_score

from sklearn.externals import joblib

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

import warnings

warnings.filterwarnings('ignore')


def load_data(database_filepath):
    '''
    This method will:
        1. Initiate DB engine.
        2. Read DisasterRecovery table from DB.
        3. Create a Pandas Data Frame from DisasterRecovery table.
        4. Split the data frame into features (in this case single string column) and target (multiple category columns).
        5. Return features, target and category name information to the calling method.

    :param database_filepath:
        File path of .db file

    :return:
        X:
            Feature data frame
        y:
            Target data frame
        category_names:
            List of message categories.
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql_table('DisasterRecovery', engine)

    category_names = df.columns.drop(['id', 'message', 'original', 'genre'])

    X = df['message']
    y = df[category_names]

    y.related.replace(to_replace=2, value=1, inplace=True)

    return X, y, category_names


def tokenize(text):
    '''
    This method will:
        1. Replace any special characters with ' '.
        2. Change to lower case and trim white spaces from prefix and suffix.
        3. Remove english stop words and split string into multiple tokens.
        4. Extract stem value from each of the tokens.
        5. Lemmatize each of the tokens.

    :param text:
        String value to be tokenize.

    :return:
        Tokens extracted from String.
    '''
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    tokens = word_tokenize(text.lower().strip())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [PorterStemmer().stem(w) for w in tokens]
    return [WordNetLemmatizer().lemmatize(w) for w in tokens]


def build_model():
    '''
    This method will create a Machine learning pipeline for training and testing. There are two steps in this pipeline:
        1. TF-IDF vectorization
        2. AdaBoostClassifier with 25 estimators.

    :return:
        Pipeline instance
    '''
    transformer = make_union(TfidfVectorizer(tokenizer=tokenize))

    # clf = AdaBoostClassifier(n_estimators=25)

    clf = AdaBoostClassifier()

    pipeline = Pipeline([
        ('transformer', transformer),
        ('classifier', MultiOutputClassifier(clf, n_jobs=-1))
    ])

    parameters = [
        {
            "classifier__estimator": [AdaBoostClassifier()],
            "classifier__estimator__base_estimator": [
                DecisionTreeClassifier(),
                MultinomialNB()],
            "classifier__estimator__n_estimators": [25, 35, 45]
        },
        {
            "classifier__estimator": [RandomForestClassifier()],
            "classifier__estimator__n_estimators": [25, 35, 45]
        }
    ]

    gridSearch = GridSearchCV(pipeline, parameters, verbose=2, n_jobs=-1)

    # return make_pipeline(transformer, MultiOutputClassifier(clf, n_jobs=-1))

    return gridSearch


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This method will:
        1. Use trained pipeline to predict output for test data.
        2. Generate Precision, Recall and F-score values for the pipeline.

    :param model:
        Pre-trained Pipeline.

    :param X_test:
        Test features

    :param Y_test:
        Test target

    :param category_names:
        List of categories

    :return:
        None
    '''
    Y_pred = model.predict(X_test)

    print(accuracy_score(y_true=Y_test, y_pred=Y_pred))

    print(classification_report(y_pred=Y_pred, y_true=Y_test, target_names=category_names))


'''
    This method will persist pre-trained pipeline as pickle file. We are persisting 2 instances of the picke file, one 
    with timestamp (to store historical versions), another without timestamp (will override existing file).
'''


def save_model(model, model_filepath):
    '''
    This method will persist pre-trained pipeline as pickle file. We are persisting 2 instances of the picke file, one
    with timestamp (to store historical versions), another without timestamp (will override existing file).

    :param model:
        Pre-trained Pipeline.

    :param model_filepath:
        File path to store pre-trained pipeline.

    :return:
        None
    '''
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    joblib.dump(model,
                model_filepath + "_{timestamp}.pkl".format(timestamp=timestamp))

    joblib.dump(model, model_filepath + ".pkl".format(timestamp=timestamp))


def main():
    '''
    This is the invocation method for this Python script.

    :return:
        None
    '''
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
