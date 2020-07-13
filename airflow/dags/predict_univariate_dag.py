from datetime import timedelta, datetime
import numpy as np
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from keras.models import load_model
from utils import read_data
from univariate_lstm import train_save_uni

def train_model():
    train_save_uni()

def load():
    model = load_model('univariate.h5')
    model.summary()
    return model

def predict():
    model = load_model('univariate.h5')

    n_steps = 24
    n_features = 1
    table = 'tsdb_cams_mera_cleaned'
    df = read_data(table)
    seq = np.array(df['consumption'])
    x_input = seq[-n_steps:]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(f'Утре очекуваме да имаме: {round(int(yhat))}W/h потрошувачка!')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
    'email': ['meto.novko@yahoo.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(seconds=10),
}

dag = DAG(
    'Predicting_Univariate',
    default_args=default_args,
    description='Modeling and predicting with univariate lstm',
    schedule_interval='@once'
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

load_model_task = PythonOperator(
    task_id='load_model',
    python_callable=load,
    dag=dag
)

prediction_task = PythonOperator(
    task_id='predict_univariate',
    python_callable=predict,
    dag=dag,
)


train_model_task >> [prediction_task]