import io
import os
import logging
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from airflow.providers.amazon.aws.hooks.s3 import S3Hook


# -----------------
# Конфигурация переменных
# -----------------
AWS_CONN_ID = "S3_CONNECTION"
MY_NAME = "Alexey"
MY_SURNAME = "Lyublinskiy"
MY_TELENICK="@Alexxx_tot"
MLFLOW_EXPERIMENT_NAME = f"{MY_SURNAME}{MY_NAME[0]}_Final"
RT=22 # RAndom STate
MLFLOW_TRACKING_URI= "http://mlflow-service:5000"
S3_BUCKET = Variable.get("S3_BUCKET")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")
AWS_ENDPOINT_URL="http://minio:9000"
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
def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)
    
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
    train_df,test_df=train_test_split(df,test_size=0.2,random_state=RT,stratify=df['survived'])
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
    


def train_and_log_mlflow(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
    AWS_ENDPOINT_URL,
    MLFLOW_TRACKING_URI,
    **context,
):
    # Установка переменных окружения
    #AAAAA убил пару часов из-за SignatureDoesNotMatch.. оказалось в переменных AirFlow неверное серкетное слово было...
    # а оно было под зведчоками... пока експорт не сделал не увидел... ААААААА
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_ENDPOINT_URL"] = AWS_ENDPOINT_URL
    os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION


    
    from sklearn.ensemble import RandomForestClassifier
    X_train = s3_read_csv(MYHOOK, S3_BUCKET, f"{MY_SURNAME}/X_train")
    y_train = s3_read_csv(MYHOOK, S3_BUCKET, f"{MY_SURNAME}/y_train")
    X_test = s3_read_csv(MYHOOK, S3_BUCKET, f"{MY_SURNAME}/X_test")
    y_test = s3_read_csv(MYHOOK, S3_BUCKET, f"{MY_SURNAME}/y_test")
    logging.info(f"Data was load from minIO")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri("http://mlflow-service:5000")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp=client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp==None:
        experiment_id = client.create_experiment(MLFLOW_EXPERIMENT_NAME)
    else:
        experiment_id = exp.experiment_id
    models={
   "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RT),
    "LogisticRegression" : LogisticRegression(max_iter=500, random_state=RT),
     "HistGB": GradientBoostingClassifier(random_state=RT)
}
    result={}
    with mlflow.start_run(run_name=MY_TELENICK, experiment_id=experiment_id) as parent_run:
        mlflow.set_tag("creator", f"{MY_NAME} {MY_SURNAME}") # экспериментировал
        for name, model in models.items():
            with mlflow.start_run(run_name=name, experiment_id=experiment_id, nested=True) as child_run:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = {
                                'accuracy': accuracy_score(y_test, y_pred),
                                'precision': precision_score(y_test, y_pred, zero_division=0),
                                'recall': recall_score(y_test, y_pred, zero_division=0),
                                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                            }
                print(f'модель: {model} with metrics {metrics}')
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                signature = infer_signature(X_test, y_pred)
                input_example = X_train.iloc[:5]
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example
                )
                result[name]={
                    "metrics" : metrics,
                    "model_uri":model_info.model_uri
                    }
        best_model_name = max(result, key=lambda x: result[x]['metrics']['f1_score'])
        best_model=result[best_model_name]

        best_model_fam=f"{best_model_name}_{MY_SURNAME}"
        reg_model=mlflow.register_model(model_uri=best_model['model_uri'], name=best_model_fam)
        client.transition_model_version_stage(name=best_model_fam,
                                            version=reg_model.version,
                                            stage="Staging")
        print(f'Лучшая модель: {best_model_name} with f1_score: {best_model["metrics"]["f1_score"]}')
        context["ti"].xcom_push(key="best_model_name", value=best_model_fam)



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
    dag_id="hw3",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops"],
) as dag:

    init_pipeline = PythonOperator(task_id="init_pipeline", python_callable=init_pipeline)
    collect_data = PythonOperator(task_id="collect_data", python_callable=collect_data)
    split_and_preprocess = PythonOperator(task_id="split_and_preprocess", python_callable=split_and_preprocess)
    train_and_log_mlflow = PythonOperator(
        task_id="train_and_log_mlflow",
        python_callable=train_and_log_mlflow,
        op_kwargs={
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "AWS_DEFAULT_REGION": "ru-central1",
            "AWS_ENDPOINT_URL": AWS_ENDPOINT_URL,
            "MLFLOW_TRACKING_URI" : MLFLOW_TRACKING_URI
        },
    )
    serve_model = BashOperator(
    task_id="serve_model",
    bash_command="""
    export PATH=$PATH:/home/airflow/.local/bin
    MODEL_NAME="{{ ti.xcom_pull(task_ids='train_and_log_mlflow', key='best_model_name') }}"
    mlflow models serve \\
        --model-uri "models:/$MODEL_NAME/Staging" \\
        --host 0.0.0.0 \\
        --port 5015 \\
        --no-conda \\
        &

    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    
    sleep 20
    curl --fail -X POST http://0.0.0.0:5015/invocations \\
        -H "Content-Type: application/json" \\
        -d '{
                    "dataframe_split": {
                        "columns": ["age", "sibsp", "parch", "fare", "sex_male", "embarked_Q", "embarked_S", "pclass_2", "pclass_3"],
                        "data": [[42.0, 0, 0, 227.525, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    }
                }'

    STATUS=$?
    echo "Prediction: $STATUS"
    
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    exit $STATUS
    """,
    params={
        "surname": MY_SURNAME
    },
    env={
        "MLFLOW_TRACKING_URI": Variable.get("MLFLOW_TRACKING_URI"),
        "AWS_ACCESS_KEY_ID": Variable.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": Variable.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_DEFAULT_REGION": Variable.get("AWS_DEFAULT_REGION"),
        "MLFLOW_S3_ENDPOINT_URL": Variable.get("AWS_ENDPOINT_URL"), 
        "AWS_S3_FORCE_PATH_STYLE": "true", 
    },
)
    clean_up=PythonOperator(task_id="cleanup", python_callable=cleanup)

init_pipeline >> collect_data >> split_and_preprocess >> train_and_log_mlflow >> serve_model >> clean_up
