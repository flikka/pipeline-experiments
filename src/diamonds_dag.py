import airflow.operators.python_operator as po
from airflow.models import DAG
from load_data import *

import datetime


def add_features_xcom(**kwargs):
    task_instance = kwargs['ti']
    dataframe_loaded = task_instance.xcom_pull(task_ids='load_data')
    
    dataframe_loaded = dataframe_loaded[:500]
    added_features_df = add_features(dataframe_loaded)
    return added_features_df    
    
def create_x_y_from_dataframe_xcom(**kwargs):
    task_instance = kwargs['ti']
    dataframe_features_added = task_instance.xcom_pull(task_ids='add_features')
    
    X, Y = create_x_y_from_dataframe(dataframe_features_added)
    return X, Y
        
def create_model_xcom(**kwargs):
    task_instance = kwargs['ti']
    X, Y = task_instance.xcom_pull(task_ids='create_XY_input')
    
    model = create_model(X, Y)
    return model
    
def score_data_xcom(**kwargs):
    task_instance = kwargs['ti']
    model = task_instance.xcom_pull(task_ids='create_model')
    X = task_instance.xcom_pull(task_ids='create_XY_input')[0]
    
    return score_data(X, model)


def persist_scores_xcom(filename, **kwargs):
    task_instance = kwargs['ti']
    dataset = task_instance.xcom_pull(task_ids='add_features')
    predictions = task_instance.xcom_pull(task_ids='score_data')    
    
    return persist_scores(dataset, predictions, filename)

def persist_performance_xcom(filename, **kwargs):
    task_instance = kwargs['ti']
    Y = task_instance.xcom_pull(task_ids='create_XY_input')[1]
    predictions = task_instance.xcom_pull(task_ids='score_data')    
    
    persist_performance(Y, predictions, filename)


args = {
    'owner': 'airflow',
    'start_date': datetime.datetime.now()
}

the_dag = DAG(dag_id='diamond_logistic_regressor', schedule_interval=None, default_args=args)

load_data_task = po.PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_kwargs={'filename':'/tmp/diamonds.csv'},
    dag=the_dag)

add_features_task = po.PythonOperator(
    task_id='add_features',
    python_callable=add_features_xcom,
    provide_context = True,
    dag=the_dag)

create_xy_task = po.PythonOperator(
    task_id='create_XY_input',
    python_callable=create_x_y_from_dataframe_xcom,
    provide_context = True,
    dag=the_dag)

create_model_task = po.PythonOperator(
    task_id='create_model',
    python_callable=create_model_xcom,
    provide_context = True,
    dag=the_dag)


score_data_task = po.PythonOperator(
    task_id='score_data',
    python_callable=score_data_xcom,
    provide_context = True,
    dag=the_dag)


persist_scores_task = po.PythonOperator(
    task_id='persist_scores',
    python_callable=persist_scores_xcom,
    provide_context = True,
    op_kwargs={'filename':"/tmp/diamonds_with_predictions.csv"},
    dag=the_dag)


persist_performance_task = po.PythonOperator(
    task_id='persist_performance',
    python_callable=persist_performance_xcom,
    provide_context = True,
    op_kwargs={'filename':"/tmp/diamonds_performance"},
    dag=the_dag)

load_data_task >> add_features_task >> create_xy_task >> create_model_task >> score_data_task >> persist_scores_task >> persist_performance_task

