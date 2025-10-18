from datetime import datetime
import io
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable

# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "S3_CONNECTION"
S3_BUCKET = Variable.get("S3_BUCKET")
MY_NAME = "Alexey"
MY_SURNAME = "Lyublinskiy"

S3_KEY_MODEL_METRICS = f"{MY_SURNAME}/model_metrics.json"
S3_KEY_PIPELINE_METRICS = f"{MY_SURNAME}/pipeline_metrics.json"
MYHOOK = S3Hook('S3_CONNECTION')

# -----------------
# Утилиты: работа с S3 через BytesIO
# -----------------
def s3_read_csv(hook: S3Hook, bucket: str, key: str) -> pd.DataFrame:
    buf = io.BytesIO()
    hook.get_conn().download_fileobj(bucket, key, buf)
    buf.seek(0)
    return pd.read_csv(buf)


def s3_write_csv(hook: S3Hook, df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    hook.get_conn().upload_fileobj(buf, bucket, key)

def med_age(df: pd.DataFrame):
    df_filled = df.copy()
    age_means =  df_filled.groupby(['sex', 'pclass'], observed=False)['age'].mean()
    for (sex, pclass), mean_age in age_means.items():
        mask = (df_filled['sex'] == sex) & (df_filled['pclass'] == pclass) & (df_filled['age'].isna())
        df_filled.loc[mask, 'age'] = mean_age
    return df_filled
# -----------------
# Таски
# -----------------


def init_pipeline(**context):
    start_ts = datetime.now().isoformat()
    logging.info(f"Запуск пайплайна: {start_ts}")
    context["ti"].xcom_push(key="pipeline_start", value=start_ts)
    # test1=context["ti"].xcom_pull(key="pipeline_start")
    # df = pd.DataFrame(
    # np.random.randn(5, 4),  # 5 строк, 4 столбца, нормальное распределение
    # columns=['A', 'B', 'C', 'D']
    # )
    # 
    # s3_write_csv(myhook,df,"s3bucket",S3_KEY_MODEL_METRICS)


def collect_data(**context):
    from sklearn.datasets import fetch_openml
    logging.info(f"Trying load dataset about titanik")
    df=fetch_openml('titanic', version=1, as_frame=True).frame
    logging.info(f"The dataset is load")
    logging.info(f"The dataset has {len(df)} raws")
    s3_write_csv(MYHOOK,df,S3_BUCKET,f"{MY_SURNAME}/rawdata")
    logging.info(f"The dataset saved on S3 in rawdata")
    context["ti"].xcom_push(key="name_raw_data", value=f"{MY_SURNAME}/rawdata")
    
def split_and_preprocess(**context):
    from sklearn.preprocessing import OneHotEncoder as OHE
    name_data=context["ti"].xcom_pull(key="name_raw_data")
    df=s3_read_csv(MYHOOK,S3_BUCKET,name_data)
    logging.info(f"Loaded dataset from minIO")
    train_df,test_df=train_test_split(df,test_size=0.2,random_state=42,stratify=df['survived'])
    columns_to_drop = ['home.dest', 'boat', 'body', 'cabin', 'ticket', 'name']
    categorical_cols = ['sex', 'embarked', 'pclass']
    numerical_cols = ['age', 'sibsp', 'parch', 'fare']
    train_df = train_df.drop(columns=columns_to_drop)
    test_df = test_df.drop(columns=columns_to_drop)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    train_df=med_age(train_df)
    test_df=med_age(test_df)
    encoder=OHE(drop='first', sparse_output=False)
    train_enc=encoder.fit_transform(train_df[categorical_cols])
    train_enc_df=pd.DataFrame(train_enc,columns=encoder.get_feature_names_out(categorical_cols))
    test_enc = encoder.transform(test_df[categorical_cols])
    test_enc_df = pd.DataFrame(
        test_enc,
        columns=encoder.get_feature_names_out(categorical_cols)
    )
    X_train = pd.concat([
    train_df[numerical_cols].reset_index(drop=True),
    train_enc_df.reset_index(drop=True)
], axis=1)

    X_test = pd.concat([
        test_df[numerical_cols].reset_index(drop=True),
        test_enc_df.reset_index(drop=True)
    ], axis=1)

    y_train = train_df['survived'].reset_index(drop=True)
    y_test = test_df['survived'].reset_index(drop=True)

    logging.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    s3_write_csv(MYHOOK,X_train,S3_BUCKET,f"{MY_SURNAME}/X_train")
    s3_write_csv(MYHOOK,X_test,S3_BUCKET,f"{MY_SURNAME}/X_test")
    s3_write_csv(MYHOOK,y_train,S3_BUCKET,f"{MY_SURNAME}/y_train")
    s3_write_csv(MYHOOK,y_test,S3_BUCKET,f"{MY_SURNAME}/y_test")
    logging.info(f"Train and test was created on minIO")
    
def train_model(**context):
    from sklearn.ensemble import RandomForestClassifier
    X_train = s3_read_csv(MYHOOK, S3_BUCKET, f"{MY_SURNAME}/X_train")
    y_train = s3_read_csv(MYHOOK, S3_BUCKET, f"{MY_SURNAME}/y_train")
    logging.info(f"Data was load from minIO")
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    start_model=datetime.now()
    context["ti"].xcom_push(key="model_start", value=start_model.isoformat())
    model.fit(X_train, y_train.values.ravel())
    end_model=datetime.now()
    context["ti"].xcom_push(key="model_end", value=end_model.isoformat())
    time_fit=(end_model - start_model).total_seconds()
    logging.info(f"Model trained successfully for {time_fit} seconds")
    context["ti"].xcom_push(key="time_fit", value=str(time_fit))
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    MYHOOK.load_file_obj(buffer, key=f"{MY_SURNAME}/train_model.pkl", bucket_name=S3_BUCKET, replace=True)
    logging.info(f"Training Model was upload to minIO")

def collect_metrics_model(**context):
    import json 
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    X_test = s3_read_csv(MYHOOK, S3_BUCKET, f"{MY_SURNAME}/X_test")
    y_test = s3_read_csv(MYHOOK, S3_BUCKET, f"{MY_SURNAME}/y_test")
    model_file = MYHOOK.get_key(f"{MY_SURNAME}/train_model.pkl", S3_BUCKET)
    buffer = io.BytesIO()
    model_file.download_fileobj(buffer)
    buffer.seek(0)
    model = joblib.load(buffer)
    
    y_pred = model.predict(X_test)
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
    }

    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    metrics_json = json.dumps(metrics, indent=2)
    MYHOOK.load_string(
        string_data=metrics_json,
        key=S3_KEY_MODEL_METRICS,
        bucket_name=S3_BUCKET,
        replace=True
    )
def collect_metrics_pipeline(**context):
    import json 
    start_pipe=datetime.fromisoformat(context["ti"].xcom_pull(key="pipeline_start"))
    end_pipe=datetime.now()
    fit_model=context["ti"].xcom_pull(key="time_fit")
  
    pipeline_metrics = {
        "pipeline_start": start_pipe.isoformat(),
        "pipeline_end": end_pipe.isoformat(),
        "pipeline_time": (end_pipe-start_pipe).total_seconds(),
        "model_training_time": fit_model,
    }
    metrics_json = json.dumps(pipeline_metrics, indent=2)
    MYHOOK.load_string(
        string_data=metrics_json,
        key=S3_KEY_PIPELINE_METRICS,
        bucket_name=S3_BUCKET,
        replace=True
    )
    logging.info(f"Pipeline is over successfully for {(end_pipe-start_pipe).total_seconds()} seconds")
    
def cleanup(**context):
    files2delete = [
        f"{MY_SURNAME}/X_train",
        f"{MY_SURNAME}/X_test", 
        f"{MY_SURNAME}/y_train",
        f"{MY_SURNAME}/y_test",
        f"{MY_SURNAME}/rawdata"
    ]
    MYHOOK.delete_objects(bucket=S3_BUCKET,keys=files2delete)
    logging.info(f"Deleted files: {files2delete}")

            
default_args = {"owner": f"{MY_NAME} {MY_SURNAME}", "retries": 1}


with DAG(

    dag_id="hw11",

    default_args=default_args,

    start_date=datetime(2025, 1, 1),

    schedule_interval="0 6 * * 2",

    catchup=False,

    tags=["mlops"],

) as dag:


    t1 = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)

    t2 = PythonOperator(task_id="collect_data", python_callable=collect_data)

    t3 = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)

    t4 = PythonOperator(task_id="train_model", python_callable=train_model)

    t5 = PythonOperator(task_id="collect_metrics_model", python_callable=collect_metrics_model)

    t6 = PythonOperator(task_id="collect_metrics_pipeline", python_callable=collect_metrics_pipeline)

    t7 = PythonOperator(task_id="cleanup", python_callable=cleanup)

    # t1 >> t2 >> t3 >> t4 >> t5 >> t6

    t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7
