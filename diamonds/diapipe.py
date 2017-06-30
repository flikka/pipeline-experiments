import pandas as pd
import uuid
import matplotlib
from io import BytesIO
matplotlib.use('Agg')
from matplotlib import pylab

from sklearn import linear_model
from sklearn import metrics 

import azure_data

def load_diamonds_from_blob(filename):
    input_blob = azure_data.download_input_blob(filename) 
    diamonds = pd.read_csv(BytesIO(input_blob.content))
    
    return diamonds

def join_blob_and_sql(blob_data, sql_data):
    return pd.concat([blob_data, sql_data])

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
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X, Y)
    
    return lin_reg

def score_data(dataset, model):
    price_predicted = model.predict(dataset)
    
    return price_predicted

def persist_scores(dataset, scores, filename):
    dataset["price_predicted"] = scores
    data_with_scores = dataset.to_csv()
    azure_data.upload_text_to_blob(data_with_scores, filename)
    
    return dataset

def persist_performance(y, y_pred, filename):
    r2_score = metrics.r2_score(y, y_pred)
    rmsd_score = metrics.mean_squared_error(y, y_pred)
    output = "R2 score: {}\nRMSD: {}".format(r2_score, rmsd_score)
    azure_data.upload_text_to_blob(output, filename + ".txt")
        
    pylab.scatter(y, y_pred)
    pylab.xlabel("Real Price")
    pylab.ylabel("Predicted Price")
    scatterIO = BytesIO()
    pylab.savefig(scatterIO)
    scatterIO.seek(0)
    azure_data.upload_bytes_to_blob(scatterIO, filename + ".png")
    return output
    
def main():
    # A series of steps that are common to many ML pipelines. 
    # Note: currently the crucial test/validation step is missing.
    
    # 1: Load data from two sources
    from_blob = load_diamonds_from_blob("diamonds.csv")
    from_sql = azure_data.load_from_azure_sql()
    
    # 2: Join data
    dataset_joined = join_blob_and_sql(from_blob, from_sql)
    
    # 3: Add features
    dataset_extra_features = add_features(dataset_joined)
    
    # 4: Create a training set, the first X diamonds,  and a test/validation set
    training_data = dataset_extra_features[:100]
    
    test_data = dataset_extra_features[73000:74000]
    
    # 5: Create target (Y) and input(X) for ML model building
    X_train, Y_train = create_x_y_from_dataframe(training_data)
    
    # 6: Build model
    model = create_model(X_train, Y_train)
    
    # 7: Predict based on model
    X_test, Y_test = create_x_y_from_dataframe(test_data)
    predictions = score_data(X_test, model)

    uid = str(uuid.uuid4())
      
    # 8: Store results
    modelinfo = model.__class__.__name__
    modelinfo = modelinfo + "-" + str(len(X_train)) + "train-"
    
    persist_scores(test_data, predictions, "results/predictions/"+ modelinfo +uid +".csv")
    print(persist_performance(Y_test, predictions, "results/performance/"+ modelinfo + uid))
    
    
if __name__=="__main__":
    main()
    
