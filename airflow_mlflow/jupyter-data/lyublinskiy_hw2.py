import os
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")
import pandas as pd
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
MY_NAME = "Alexey"
MY_SURNAME = "Lyublinskiy"
MY_TELENICK="@Alexxx_tot"
EXPERIMENT_NAME = f"{MY_SURNAME}_{MY_NAME[:1]}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
RT=22 # RAndom STate

def med_age(df: pd.DataFrame):
    df_filled = df.copy()
    age_means =  df_filled.groupby(['sex', 'pclass'], observed=False)['age'].mean()
    for (sex, pclass), mean_age in age_means.items():
        mask = (df_filled['sex'] == sex) & (df_filled['pclass'] == pclass) & (df_filled['age'].isna())
        df_filled.loc[mask, 'age'] = mean_age
    return df_filled

def prepare_data():
    df = fetch_openml('titanic', version=1, as_frame=True).frame
    df['survived'] = df['survived'].astype(int)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['survived'])
    
    columns_to_drop = ['home.dest', 'boat', 'body', 'cabin', 'ticket', 'name']
    train_df = train_df.drop(columns=columns_to_drop)
    test_df = test_df.drop(columns=columns_to_drop)


    train_df = med_age(train_df)
    test_df = med_age(test_df)


    embarked_mode = train_df['embarked'].mode()[0]
    train_df['embarked'].fillna(embarked_mode, inplace=True)
    test_df['embarked'].fillna(embarked_mode, inplace=True)

    fare_median = train_df['fare'].median()
    train_df['fare'].fillna(fare_median, inplace=True)
    test_df['fare'].fillna(fare_median, inplace=True)

    categorical_cols = ['sex', 'embarked', 'pclass']
    numerical_cols = ['age', 'sibsp', 'parch', 'fare']


    encoder = OneHotEncoder(drop='first', sparse_output=False)
    train_enc = encoder.fit_transform(train_df[categorical_cols])
    test_enc = encoder.transform(test_df[categorical_cols])

    train_enc_df = pd.DataFrame(train_enc, columns=encoder.get_feature_names_out(categorical_cols))
    test_enc_df = pd.DataFrame(test_enc, columns=encoder.get_feature_names_out(categorical_cols))

    X_train = pd.concat([train_df[numerical_cols].reset_index(drop=True),
                         train_enc_df.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_df[numerical_cols].reset_index(drop=True),
                        test_enc_df.reset_index(drop=True)], axis=1)

    y_train = train_df['survived'].reset_index(drop=True)
    y_test = test_df['survived'].reset_index(drop=True)

    return X_train, y_train, X_test, y_test

def train_and_log(model, X_train, y_train, X_test, y_test):
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

    # в минИО все складывается, а вот в артефактах в UI почему то долго грузится и не отображается..
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    return metrics, model_info
    
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp=client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp==None:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
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
        X_train, y_train, X_test, y_test= prepare_data()
        for name, model in models.items():
            with mlflow.start_run(run_name=name, experiment_id=experiment_id, nested=True) as child_run:
                metrics, model_info =train_and_log(model, X_train, y_train, X_test, y_test)
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
                                            stage="staging")
        print(f'Лучшая модель: {best_model_name} with f1_score: {best_model["metrics"]["f1_score"]}')
if __name__ == "__main__":
    main()
