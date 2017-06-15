import pandas as pd
import matplotlib.pylab as pylab
from sklearn import linear_model
import airflow.operators.python_operator as po

def load_data(filename):
    diamonds = pd.read_csv(filename)
    print(diamonds.describe())
    return diamonds

def add_features(dataset):
    dataset["volume"] = dataset["x"] * dataset["y"] * dataset["z"]
    return dataset
    
def create_training_target(dataset):
    price = dataset["price"]
    dataset.drop("price", axis=1, inplace=True)
    dataset.drop("clarity", axis=1, inplace=True)
    dataset.drop("cut", axis=1, inplace=True)
    dataset.drop("color", axis=1, inplace=True)
    return dataset, price

def create_model(X, Y):
    log_reg = linear_model.LogisticRegression()
    log_reg.fit(X, Y)
    return log_reg

def score_data(dataset, model):
    price_predicted = model.predict(dataset)
    return price_predicted

def main():
    filename = "../data/diamonds.csv"
    dataset = load_data(filename)
    dataset = dataset[:500]
    dataset = add_features(dataset)
    X, Y = create_training_target(dataset)
    
    model = create_model(X, Y)
    predictions = score_data(X, model)
    
    pylab.scatter(Y, predictions)
    
    
if __name__=="__main__":
    main()