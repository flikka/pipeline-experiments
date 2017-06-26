import pandas as pd
import requests
from io import StringIO
import uuid

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab

from sklearn import linear_model
from sklearn import metrics 


def load_data(url):
    response = requests.get(url, verify=False)
    diamonds = pd.read_csv(StringIO(response.text))
    print("Initial dataset, summary:")
    print(diamonds.describe())
    return diamonds

def add_features(dataset):
    dataset["volume"] = dataset["x"] * dataset["y"] * dataset["z"]
    return dataset
    
def create_x_y_from_dataframe(dataset):
    price = dataset["price"]
    X = dataset.drop("price", axis=1)
    X.drop("clarity", axis=1, inplace=True)
    X.drop("cut", axis=1, inplace=True)
    X.drop("color", axis=1, inplace=True)
    return X, price

def create_model(X, Y):
    log_reg = linear_model.LogisticRegression()
    log_reg.fit(X, Y)
    return log_reg

def score_data(dataset, model):
    price_predicted = model.predict(dataset)
    return price_predicted

def persist_scores(dataset, scores, filename):
    dataset["price_predicted"] = scores
    dataset.to_csv(filename)
    return dataset

def persist_performance(y, y_pred, filename):
    r2_score = metrics.r2_score(y, y_pred)
    rmsd_score = metrics.mean_squared_error(y, y_pred)
    
    output = "R2 score: {}\nRMSD: {}".format(r2_score, rmsd_score)

    with open(filename, "w") as text_file:
        text_file.write(output)
        
    pylab.scatter(y, y_pred)
    
    pylab.xlabel("Real Price")
    pylab.ylabel("Predicted Price")
    pylab.savefig(filename + ".png")
    
def main():
    url = "https://raw.githubusercontent.com/flikka/pipeline-experiments/master/data/diamonds.csv"
    dataset = load_data(url)[:500]
    dataset = add_features(dataset)
    X, Y = create_x_y_from_dataframe(dataset)
    model = create_model(X, Y)
    predictions = score_data(X, model)

    uid = str(uuid.uuid4())
    
    filename_output = "/tmp/diamonds_predictions_from_main-"+uid +".csv"
    persist_scores(dataset, predictions, filename_output)
    persist_performance(Y, predictions, "/tmp/diamonds_performance_main-" + uid)
    
    
if __name__=="__main__":
    main()
    
