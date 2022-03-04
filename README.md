# Disaster Response Pipeline Project
This project uses data from messages to build, train, and evaluate a ML algorithm and pipeline to classify text into one or more of 36 categories around disaster response. It also serves as a web app to allow users to enter text and see how the algorithm performs. 

### Files:
process_data.py: This file serves as an ETL pipeline for data from the .csvs included to ready it for the machine learning pipeline.
train_classifier.py: This file codes the relevant code for the building, training, and evaluating of the model. 
run.py: This file loads the data from the database, creates graphs to be displayed on the frontend, and uses the model to evaluate text from user input.

classifier.pkl: saved model

master.html: main page of web app
go.html: classification result page of web app

disaster_categories.csv: data to process
disaster_messages.csv: data to process

DisasterResponse.db: database to save clean data to


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
