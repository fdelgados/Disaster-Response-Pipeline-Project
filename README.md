# Disaster Response Pipeline Project

A machine learning pipeline to categorize emergency messages based on the needs communicated by the sender.

### Intallations:

Apart from the libraries installed with the Anaconda distribution, you need to install the plotly python library. To do that, run the following commands:

`$ pip install plotly`

or

`$ conda install -c plotly plotly`


### Project Motivation:

The Disaster Response Pipeline Project aims to be a useful tool for emergency services in a disaster scenario
optimizing the resources and accelerating the response in humanitarian crises (earthquakes, floods, extreme weather events, etc).
The project is composed of three parts:

1. An ETL pipeline, that Extracts disaster-related messages classified in 36 categories (listed below), Transforms the data through a wrangling process
so that they can be used to train a Machine Learning model, and saves (Loads) the data into a database.
2. A Machine Learning pipeline. In this part the data is supplied to a model to be trained using natural language process techniques.
The trained model is saved in a file that will be used to classify new messages.
3. A web application in which a user can enter messages, which will be classified according to the mentioned categories. 

#### Categories

- Related
- Request
- Offer
- Aid Related
- Medical Help
- Medical Products
- Search And Rescue
- Security
- Military
- Water
- Food
- Shelter
- Clothing
- Money
- Missing People
- Refugees
- Death
- Other Aid
- Infrastructure Related
- Transport
- Buildings
- Electricity
- Tools
- Hospitals
- Shops
- Aid Centers
- Other Infrastructure
- Weather Related
- Floods
- Storm
- Fire
- Earthquake
- Cold
- Other Weather
- Direct Report

### File Descriptions:

The most important files are listed below:

- `data/process_data.py`: this file contains the code to perform the ETL pipeline. 
It receives as arguments the data files containing the categorized messages and categories, and the path to the final database file.
- `models/train_classifier.py`: this file contains the code to perform the Machine Learning (NLP) pipeline. It receives the transformed data database
and the path to the trained model file as arguments.
- `app/run.py`: this is the web app main file.

See the Instructions section for more info.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/


### License

Copyright 2019 Francisco Delgado

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### References

- Raschka S. & Mirjalili V. (2017) *Python Machine Learning*. Packt Publishing Ltd.
- Bengford B., Bilbro R. & Ojeda T. (2018) *Applied Text Analysis with Python*. O'Reilly Media Inc.
