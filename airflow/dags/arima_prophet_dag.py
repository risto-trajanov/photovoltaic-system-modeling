from datetime import timedelta, datetime

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

import preprocessing
#import prophet
import arima


def process_data():
    preprocessing.main()


# def train_prophet():
#     prophet.main()


def train_arima():
    arima.main()


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['meto.novko@yahoo.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    #'retries': 2,
    #'retry_delay': timedelta(seconds=10),
}

dag = DAG(
    'Arima_Prophet_Modeling',
    default_args=default_args,
    description='Preprocessing and modeling data',
    schedule_interval='@once'
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=process_data,
    dag=dag
)

# prophet_task = PythonOperator(
#     task_id='train_prophet',
#     python_callable=train_prophet(),
#     dag=dag
# )

arima_task = PythonOperator(
    task_id='train_arima',
    python_callable=train_arima,
    dag=dag
)

preprocess_task >> [arima_task]
