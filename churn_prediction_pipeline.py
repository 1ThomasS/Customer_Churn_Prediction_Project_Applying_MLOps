import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, col, datediff, current_date, unix_timestamp
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(spark, input_path):
    """Preprocess the data."""
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # Drop columns with >60% null values
    columns_to_drop = [
        'sub_industry_id', 'sub_industry_name', 'standardized_corp_level', 'introducer', 'company_vip', 'active_date'
    ]
    df = df.drop(*columns_to_drop)

    # Filter individual data
    df = df.filter((df["sector_name"] == "Ca nhan") & (df["industry_name"] == "X0000. Tu nhan"))

    # Drop irrelevant columns
    irrelevant_columns = [
        'sector_id', 'sector_name', 'customer_level', 'customer_level_change', 'mis_date', 'record_end_date',
        'lob_code_map', 'industry_code', 'industry_id', 'industry_name', 'target_id', 'branch_code',
        'business_segment', 'sub_customer_level_change', 'short_name', 'full_name', 'staff_indicator',
        'account_officer', 'company_name', 'main_business_class_code'
    ]
    df = df.drop(*irrelevant_columns)

    # Handle missing values
    df = df.fillna({
        "customer_joining_age": df.select(mean("customer_joining_age")).first()[0],
        "customer_age": df.approxQuantile("customer_age", [0.5], 0.01)[0],
        "gender": df.groupBy("gender").count().orderBy("count", ascending=False).first()["gender"],
        "business_class_code": df.groupBy("business_class_code").count().orderBy("count", ascending=False).first()[
            "business_class_code"],
        "tenure_years": df.approxQuantile("tenure_years", [0.5], 0.25)[0],
        "customer_segment": df.groupBy("customer_segment").count().orderBy("count", ascending=False).first()[
            "customer_segment"]
    })

    # Convert date columns to numeric
    date_columns = [
        "customer_open_date", "customer_start_date", "record_start_date",
        "contact_date", "contact_time", "segment_start_date", "segment_end_date"
    ]
    for date_col in date_columns:
        if "DATE_TIME" in date_col:
            df = df.withColumn(f"{date_col}_numeric", unix_timestamp(col(date_col)))
        else:
            df = df.withColumn(f"{date_col}_numeric", datediff(current_date(), col(date_col)))

    df = df.drop(*date_columns)

    return df


def feature_engineering(df):
    """Perform feature engineering."""
    # Encode categorical variables
    categorical_columns = [
        "target_name", "customer_segment", "business_class_code", "gender", "lifecycle_stage", "customer_status"
    ]
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in categorical_columns]
    encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded") for col in categorical_columns]

    pipeline = Pipeline(stages=indexers + encoders)
    df_transformed = pipeline.fit(df).transform(df)

    # Select final features
    encoded_columns = [f"{col}_encoded" for col in categorical_columns]
    numeric_columns = [
        "customer_joining_age", "marital_status", "customer_age",
        "tenure_years", "customer_open_date_numeric", "customer_start_date_numeric",
        "record_start_date_numeric", "contact_date_numeric", "contact_time_numeric",
        "segment_start_date_numeric", "segment_end_date_numeric"
    ]
    final_columns = encoded_columns + numeric_columns + ["churn"]

    df_final = df_transformed.select(*final_columns)

    # Scale features
    assembler = VectorAssembler(inputCols=final_columns[:-1], outputCol="all_features")
    scaler = StandardScaler(inputCol="all_features", outputCol="scaled_features", withMean=True, withStd=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    scaled_model = pipeline.fit(df_final)
    df_scaled = scaled_model.transform(df_final)

    return df_scaled.select("scaled_features", "churn")


def train_models(df_scaled):
    """Train Logistic Regression, Random Forest, and XGBoost models."""
    # Split data
    train_data, test_data = df_scaled.randomSplit([0.8, 0.2], seed=42)

    # Logistic Regression
    lr = LogisticRegression(labelCol="churn", featuresCol="scaled_features")
    lr_model = lr.fit(train_data)
    lr_predictions = lr_model.transform(test_data)

    # Random Forest
    rf = RandomForestClassifier(labelCol="churn", featuresCol="scaled_features", seed=42)
    rf_model = rf.fit(train_data)
    rf_predictions = rf_model.transform(test_data)

    # XGBoost
    train_pd = train_data.toPandas()
    test_pd = test_data.toPandas()
    x_train = np.array([row['scaled_features'].toArray() for row in train_pd.to_dict('records')])
    y_train = train_pd["churn"].values
    x_test = np.array([row['scaled_features'].toArray() for row in test_pd.to_dict('records')])
    y_test = test_pd["churn"].values

    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(x_train, y_train)

    return lr_model, rf_model, xgb_model, test_data, x_test, y_test


def evaluate_model(model, predictions, x_test, y_test, name):
    """Evaluate the models."""
    if name == "XGBoost":
        y_pred = model.predict(x_test)
    else:
        y_pred = predictions.select("prediction").collect()

    print(f"{name} Evaluation Complete!")


def main():
    spark = SparkSession.builder.appName("ChurnPredictionPipeline").getOrCreate()
    input_path = "path_to_input_data.csv"

    # Preprocessing
    df = preprocess_data(spark, input_path)

    # Feature Engineering
    df_scaled = feature_engineering(df)

    # Train Models
    lr_model, rf_model, xgb_model, test_data, x_test, y_test = train_models(df_scaled)

    # Evaluate Models
    evaluate_model(lr_model, None, x_test, y_test, "Logistic Regression")
    evaluate_model(rf_model, None, x_test, y_test, "Random Forest")
    evaluate_model(xgb_model, None, x_test, y_test, "XGBoost")


if __name__ == "__main__":
    main()
