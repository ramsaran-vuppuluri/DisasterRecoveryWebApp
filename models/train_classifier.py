'''
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

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report

from sklearn.externals import joblib

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

import warnings

warnings.filterwarnings('ignore')


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql_table('InsertTableName', engine)

    category_names = df.columns.drop(['id', 'message', 'original', 'genre'])

    X = df['message']
    y = df[category_names]

    y.related.replace(to_replace=2, value=1, inplace=True)

    return X, y, category_names


def tokenize(text):
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    tokens = word_tokenize(text.lower().strip())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [PorterStemmer().stem(w) for w in tokens]
    return [WordNetLemmatizer().lemmatize(w) for w in tokens]


def build_model():
    transformer = make_union(TfidfVectorizer(tokenizer=tokenize))

    clf = AdaBoostClassifier(n_estimators=25)

    return make_pipeline(transformer, MultiOutputClassifier(clf, n_jobs=-1))


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    print(classification_report(y_pred=Y_pred, y_true=Y_test, target_names=category_names))


def save_model(model, model_filepath):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    version = joblib.__version__

    joblib.dump(model, model_filepath + "_{version}_{timestamp}.pkl".format(version=version, timestamp=timestamp))

    joblib.dump(model, model_filepath + "_{version}.pkl".format(version=version, timestamp=timestamp))


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
