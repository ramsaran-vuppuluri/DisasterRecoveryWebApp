# Disaster Response Pipeline Project

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. I have created a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Project Components:

There are three components in this project.

1. ETL Pipeline
   
    __process_data.py__ Python script that:
    
    * Loads the messages and categories datasets
    * Merges the two datasets
    * Cleans the data
    * Stores it in a SQLite database

2. ML Pipeline

    __train_classifier.py__ Python script that:
    
    * Loads data from the SQLite database
    * Splits the dataset into training and test sets
    * Builds a text processing and machine learning pipeline
    * Trains and tunes a model using GridSearchCV
    * Outputs results on the test set
    * Exports the final model as a pickle file
    
3. Flask Web App   

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python3 process_data.py disaster_messages.csv disaster_categories.csv DisasterRecovery.db`
    - To run ML pipeline that trains classifier and saves
        `python3 train_classifier.py ../data/DisasterRecovery.db AdaBoostClassifier`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Libraries
    
    | Library       | Version   |
    | ------------- |-----------|
    | Flask         | 0.12.2    |
    |sqlalchemy     | 1.2.1     |
    |Pandas         | 0.22.0    |
    |nltk           | 3.2.5     |
    |sklearn        | 0.19.1    |