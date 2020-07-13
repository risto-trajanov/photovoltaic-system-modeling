import tempfile
from datetime import datetime, timedelta

import pandas
from airflow.hooks.S3_hook import S3Hook
from airflow.hooks.postgres_hook import PostgresHook
from sqlalchemy import create_engine


def get_connection():
    # hook = PostgresHook(conn_name_attr='tsdb')
    # conn = hook.get_connection('tsdb')
    SQLALCHEMY_DATABASE_URI = 'postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}'.format(
        db_host = '34.69.215.94',
        db_name = 'snpiao_data',
        db_password = 'ucamnafinki',
        db_user = 'snpiao_data',
    )

    # db_host = '34.69.215.94',
    # db_name = 'snpiao_data',
    # db_password = 'ucamnafinki',
    # db_user = 'snpiao_data',

    # db_host = conn.host,
    # db_name = conn.schema,
    # db_password = conn.password,
    # db_user = conn.login,

    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    return engine.connect()


def read_data(table):
    connection = get_connection()
    data = pandas.read_sql_table(table, connection)
    data = data.set_index('datetime')
    connection.close()
    return data


def read_data_for_day(table, day, id=False):
    connection = get_connection()

    day = datetime.strptime(day, '%Y-%m-%d')  # YYYY-MM-DD

    day.replace(hour=0, minute=0, second=0)
    query = 'SELECT * FROM {table} WHERE "datetime" >= \'{day}\' and "datetime" <=\'{next_day}\''. \
        format(table=table, day=day, next_day=day + timedelta(days=1))

    data = pandas.read_sql_query(query, connection)
    data = data.set_index('datetime')
    #del data['id']
    connection.close()
    return data


def persist_data(table, data, id=False):
    data.reset_index(inplace=True)
    if id:
        data['id'] = id

    if 'index' in data:
        del data['index']

    try:
        min = data['datetime'].min()
        max = data['datetime'].max()
    except:
        return

    if not isinstance(min, pandas.Timestamp):
        raise Exception()

    connection = get_connection()
    trans = connection.begin()
    try:
        deleteToUpdate = connection.execute(
            'DELETE FROM {table} WHERE "datetime" >= \'{day}\' and "datetime" <=\'{next_day}\''.format(day=min,
                                                                                                       next_day=max,
                                                                                                       table=table))

        data.to_sql(name=table, con=connection,
                    if_exists="append",
                    index=False,
                    method='multi'
                    )
        trans.commit()
    except Exception as e:
        print('rollback')
        trans.rollback()
        connection.close()
        raise e

    connection.close()

    return


S3_BUCKET_NAME = 'snpiao'


# def download_s3(file):
#     s3 = S3Hook()
#     with tempfile.NamedTemporaryFile(delete=False) as tmp:
#         s3.get_key(file, S3_BUCKET_NAME).download_fileobj(tmp)
#         return tmp.name
#
#
# def upload_s3(file):
#     s3 = S3Hook()
#     return s3.load_file(file, file, bucket_name=S3_BUCKET_NAME, replace=True)
