# Project setup

## Install project dependencies
The `requirements.txt` file contains all dependencies of the project. Install them using pip to set up the project: `pip install -r requirements.txt`. Although optional, it's recommended to install the dependencies in a virtual environment (I used venv).

## Running the Python script
In your terminal, run `python script.py train.csv test.csv`

Note: 
* Write the two filenames with their extensions (.csv, in the example above), starting with the training dataset.

## Viewing the model's predictions
After running the Python script, the predictions will be written by default to the file `model_predictions.txt` in the same directory as the script.