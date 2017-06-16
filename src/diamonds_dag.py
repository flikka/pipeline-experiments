import airflow.operators.python_operator as po
from airflow.models import DAG
from load_data import load_data

import datetime

args = {
    'owner': 'airflow',
    'start_date': datetime.datetime.now()
}


the_dag = DAG(dag_id='diamond_logistic_regressor', schedule_interval=None, default_args=args)
parameters = {} #Used to pass objects between tasks

load_data_task = po.PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_kwargs={'filename':'/tmp/diamonds.csv'},
    dag=the_dag)

#Next task, use the context with XCom to pick up the data returne from task 1
