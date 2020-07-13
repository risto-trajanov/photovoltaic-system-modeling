from datetime import timedelta
from datetime import datetime
import pandas as pd
# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import arima
import preprocessing
import crawl_features_weatherbit
import sma_crawl as sma
import prediction
import utils


lat = 51.1877
long = 10.0398
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}


def sma_crawl_function():
    sma.main()


def weatherbit_crawl_function():
    crawl_features_weatherbit.main()


def process_data_function():
    preprocessing.main()


# def train_prophet():
#     prophet.main()
#

def train_arima_function():
    arima.main()


def predictions_function():
    prediction.main()


dag = DAG(
    'Energize',
    default_args=default_args,
    description='A simple tutorial DAG',
    schedule_interval=timedelta(days=1),
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=process_data_function,
    dag=dag
)

# prophet_task = PythonOperator(
#     task_id='train_prophet',
#     python_callable=train_prophet(),
#     dag=dag
# )

arima_task = PythonOperator(
    task_id='train_arima',
    python_callable=train_arima_function,
    dag=dag
)

sma_crawl_task = PythonOperator(
    task_id='sma_crawl',
    python_callable=sma_crawl_function,
    dag=dag
)

weatherbit_crawl_task = PythonOperator(
    task_id='weatherbit_crawl',
    python_callable=weatherbit_crawl_function,
    dag=dag
)

prediction_task = PythonOperator(
    task_id='predictions',
    python_callable=predictions_function,
    dag=dag
)

[sma_crawl_task, weatherbit_crawl_task] >> preprocess_task >> arima_task >> prediction_task
