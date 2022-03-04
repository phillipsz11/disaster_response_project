import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle

import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

def load_data(database_filepath):
    """Method to load data from database.
        
        Args: 
            database_filepath: path to the database file
        
        Returns: 
            X: messages of the data set
            Y: categories of messages
            labels: the list of categories for the messages
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disaster', con=engine)
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    labels = ['related', 'request', 'offer', 'aid_related', 'medical_help',
        'medical_products', 'search_and_rescue', 'security', 'military',
        'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
        'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
        'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops',
        'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm',
        'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    return X, Y, labels


def tokenize(text):
    """Method to tokenize text from messages.
        
        Args: 
            text: the string message
        
        Returns: 
            clean_tokens: lemmatized, normalized words of the message
    """
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """Method to build ML pipeline for model.
        
        Args: 
            None
        
        Returns: 
            model: the fitted model
    """
    pipeline_ada = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(n_estimators=60)))
    ])


    model = pipeline_ada
    
    return model

    


def evaluate_model(model, X_test, Y_test, category_names):
    """Method to evaluate model performance on test set.
        
        Args: 
            model: the fitted model
            X_test: the test set used for predictions
            Y_test: the test set used to show classification report
            category_names: labels used to show how well model performed for each category
        
        Returns: 
            None
    """
    y_pred = model.predict(X_test)
    
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Method to save model.
        
        Args: 
            model: the fitted model
            model_filepath: the file path to save the model to
        
        Returns: 
            None
    """
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()