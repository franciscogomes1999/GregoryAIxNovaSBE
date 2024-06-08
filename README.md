# GregoryAIxNovaSBE

# GregoryAIxNovaSBE

## Classification of Relevant Multiple Sclerosis Articles

### Project Description
Our goal is to enhance the existing Machine Learning solution of GregoryAI with the overall aim to facilitate more accurate knowledge sharing among multiple sclerosis patients, medical staff, and researchers. We have implemented [insert here model name] model with a [insert here relevant metrics].

### Installation
1. Clone the repository

git clone https://github.com/franciscogomes1999/GregoryAIxNovaSBE.git

2. Install required packages

pip install -r requirements.txt

### Usage
##### Train and Tune the Model
1. Open and run the train_tune_notebook.ipynb notebook.
2. Follow the instructions within the notebook to train and tune the model.

##### Classify New Articles
1. Open and run the classify_notebook.ipynb notebook.
2. Follow the instructions within the notebook to classify new articles.


### Pipelines
##### Train & Tune Pipeline
1.  Data: Retrieve the articlesdataset.csv from the database.
2 Unprocessed Data: The raw dataset is saved as articlesdataset.csv.
3. Clean + Preprocess: Data cleaning and preprocessing steps are performed.
4. Processed Data: The cleaned data is saved as processed_data.csv.
5. Split Data: Split the data into training, validation, test, and unlabelled datasets.
6. Pseudo Labeling:
- Generate pseudo labels for the unlabelled data.
- Filter and select high-confidence pseudo labels.
7. Train:
- Train the model using the labelled and pseudo-labelled data.
- Store the trained model weights.
8. Evaluate: Evaluate the model's performance using validation and test datasets.

![Train & Tune Pipeline](images/train_tune_pipeline.jpg)



##### Classify Pipeline
1. Download Data: Retrieve the articlesdataset.csv from the database.
2. Unprocessed Data: The raw dataset is saved as articlesdataset.csv.
3. Clean + Preprocess: Data cleaning and preprocessing steps are performed.
4. Filtered and Processed Data: The cleaned data is saved as new_unlabelled_articles.csv.
5. Classify:
- Import the model weights.
- Classify the articles to generate predictions.
6. Output: Save the updated CSV file with predicted labels for new articles.

![Classify Pipeline](images/classify_pipeline.jpg) 


### Configuration
The configuration settings for the project can be found in the config.yaml file. Modify this file to change parameters such as data paths, model hyperparameters, and training settings.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact
If you have any questions or feedback, please contact one of the following contacts:
- Julia Antonioli - 60178@novasbe.pt
- Kuba Bialczyk - 61678@novasbe.pt
- Nicolò Mazzoleni - 59935@novasbe.pt
- Francisco Gomes - 39350@novasbe.pt
- Martim Esteves - 46953@novasbe.pt