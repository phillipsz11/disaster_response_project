import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Method to load data from csv and merge.
        
        Args: 
            messages_filepath: file path to the messages dataset
            categories_filepath: file path to the categories dataset
        
        Returns: 
            df: merged DataFrame of messages and categories
            categories: DataFrame to be used in the cleaning process
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df, categories


def clean_data(df, categories):
    """Method to clean the merged DataFrame.
        
        Args: 
            df: the merged DataFrame to be cleaned
            categories: DataFrame used in the cleaning process
        
        Returns: 
            df: cleaned dataframe
    
    """
    categories_fixed = categories['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories_fixed.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.str[:-2]).values
    
    # rename the columns of `categories`
    for category in category_colnames:
        categories_fixed.columns = category
    
    for column in categories_fixed:
        # set each value to be the last character of the string
        categories_fixed[column] = categories_fixed[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories_fixed[column] = categories_fixed[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_fixed], axis=1)

    #drop nulls from the category columns
    df.dropna(subset=['related', 'request', 'offer', 'aid_related', 'medical_help',
        'medical_products', 'search_and_rescue', 'security', 'military',
        'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
        'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
        'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops',
        'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm',
        'fire', 'earthquake', 'cold', 'other_weather', 'direct_report'], inplace=True)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    #replace the 2 values in the dataframe with 1 for the ML model
    df.replace(2,1, inplace=True)
    
    #convert category columns from float to int
    df[['related', 'request', 'offer', 'aid_related', 'medical_help',
        'medical_products', 'search_and_rescue', 'security', 'military',
        'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
        'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
        'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops',
        'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm',
        'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']] = df[['related', 'request', 'offer', 'aid_related', 'medical_help',
        'medical_products', 'search_and_rescue', 'security', 'military',
        'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
        'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related',
        'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops',
        'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm',
        'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']].apply(np.int64)
    
    return df


def save_data(df, database_filename):
    """Method to load cleaned dataframe into database.
        
        Args: 
            df: the dataframe to be loaded
            database_filename: the name of the database file
        
        Returns: 
            None
    
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql('Disaster', engine, if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
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