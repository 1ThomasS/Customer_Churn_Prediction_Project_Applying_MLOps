import os
import pandas as pd
import argparse
from pathlib2 import Path
import pyspark
from pyspark.sql import SparkSession

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='fetch data args')
    parser.add_argument('-b', '--base_path',
                        help='directory to base', default='../')
    parser.add_argument(
        '-d', '--data', help='directory to training data', default='train')
    parser.add_argument(
        '-t', '--target', help='target file to hold good data', default='train.csv') 
   
    # parse arguments   
    args = parser.parse_args()
    
    # define paths
    base_path = Path(args.base_path).resolve(strict=False)
    data_path = base_path.joinpath(args.data).resolve(strict=False)
    target_path = Path(data_path).resolve(strict=False).joinpath(args.target)
    print('Train File: {}'.format(target_path))
    
    # read input csv file
    spark = SparkSession.builder.appName('Churn_Prediction').getOrCreate()

    df_1 = spark.read.parquet("DC_TEST.parquet")
    df_2 = spark.read.parquet('DIM_CUSTOMER_20240922.parquet')
    df_2 = df_2.withColumnRenamed('V_D_CUST_REF_CODE', 'CIF')

    df = df_1.join(df_2, on='CIF', how='inner')

    column_mapping = {
    'CIF': "customer_id",
    'SHORT_NAME': "short_name",
    "CUS_OPEN_DATE": "customer_open_date",
    "SECTOR_ID": "sector_id",
    "SECTOR_NAME": "sector_name",
    "INDUSTRY_ID": "industry_id",
    "INDUSTRY_NAME": "industry_name",
    "SUB_INDUSTRY_ID": "sub_industry_id",
    "SUB_INDUSTRY_NAME": "sub_industry_name",
    "TARGET_ID": "target_id",
    "TARGET_NAME": "target_name",
    "ACOUNT_OFFICER": "account_officer",
    "COMPANY_BOOK": "company_book",
    "COMPANY_NAME": "company_name",
    "PKKH": "customer_level",
    "PKKH_DIEU_CHUYEN": "customer_level_change",
    "PKKH_DIEU_CHUYEN_SUB": "sub_customer_level_change",
    "ABB_STAND_CORP": "standardized_corp_level",
    "BUCKET_ABB": "bucket_abb",
    "INTRODUCER": "introducer",
    "COMPANY_VIP": "company_vip",
    "MIS_DATE": "mis_date",
    "N_D_CUST_JOINING_AGE": "customer_joining_age",
    "V_D_CUST_MARITAL_STATUS": "marital_status",
    "N_D_CUST_AGE": "customer_age",
    "V_D_CUST_GENDER": "gender",
    "F_D_CUST_STAFF_IND": "staff_indicator",
    "D_D_CUST_START_DATE": "customer_start_date",
    "V_D_CUST_BUSS_SEGMENT": "business_segment",
    "V_D_CUST_BRANCH_CODE": "branch_code",
    "V_D_INDUSTRY_CODE": "industry_code",
    "V_D_CUST_SEGMENT": "customer_segment",
    "D_RECORD_START_DATE": "record_start_date",
    "D_RECORD_END_DATE": "record_end_date",
    "V_CUSTOMER_FULL_NAME": "full_name",
    "AV_BUSINESS_CLASS_CODE": "business_class_code",
    "AV_MAIN_BUSINESS_CLASS_CODE": "main_business_class_code",
    "AD_CONTACT_DATE": "contact_date",
    "AD_DATE_TIME": "contact_time",
    "AV_LOB_CODE_MAP": "lob_code_map",
    "F_ACTIVE_FLAG": "churn",
    "D_ACTIVE_DATE": "active_date",
    "V_CUSTOMER_STATUS": "customer_status",
    "D_SEGMENT_START_DATE": "segment_start_date",
    "D_SEGMENT_END_DATE": "segment_end_date",
    'ACCOUNT_OFFICER': 'account_officer',

    }
    for old_name, new_name in column_mapping.items():
        df = df.withColumnRenamed(old_name, new_name)
    print(df.to_string())

    # create path if it doesn't exist
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Save processed input to a new csv file
    df.write.mode('overwrite').parquet(target_path, index=False)

