import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

df = pd.read_csv('data_science_salaries.csv')
df['emp_is_UnitedStates'] = np.where(df['employee_residence']=='United States',1,0)
df=df.drop(columns='emp_is_UnitedStates')
df_encoded = pd.get_dummies(df , columns=['work_models'], drop_first=True, dtype=int)
df_encoded = df_encoded.drop(columns='employee_residence')

df_encoded['comp_is_US'] = np.where(df_encoded['company_location']=='United States',1,0)
df_encoded=df_encoded.drop(columns='company_location')

column_name = 'company_size'
size_mapping = {
    'Small': 0,
    'Medium': 1,
    'Large': 2
}
df_encoded['company_size_encoded'] = df_encoded[column_name].map(size_mapping)
df_encoded=df_encoded.drop(columns='company_size')

salaryTemp = df_encoded['salary_in_usd']
salaryTemp = np.log(salaryTemp)
df_encoded['salary_in_usd'] = salaryTemp.round(3)

def jobClassify(title):
    #
    title = str(title).lower()

    ds_keywords = ['scientist', 'science', 'research', 'nlp', 'deep learning']
    if any(k in title for k in ds_keywords):
        return 'Data Science'
    
    ml_keywords = ['ml', 'machine learning', 'ai', 'mlops', 'computer vision']
    if any(k in title for k in ml_keywords):
        return 'Machine Learning / AI'

    de_keywords = ['engineer', 'infrastructure', 'etl', 'architect', 'developer']
    if any(k in title for k in de_keywords):
        return 'Data Engineering'
        
    da_keywords = ['analyst', 'bi', 'analytics', 'insights', 'visualization'] 
    if any(k in title for k in da_keywords):
        return 'Data Analysis'
    
    return 'Other'

df_encoded['domain_expertise'] = df_encoded['job_title'].apply(jobClassify)
df_encoded=df_encoded.drop(columns='job_title')
df_encoded = pd.get_dummies(df_encoded, columns=['domain_expertise'], drop_first=True, dtype=int)
print(df_encoded.head())

column_name2='experience_level'
exp_mapping = {
    'Entry-level': 0,
    'Senior-level': 1,
    'Mid-level': 2,
    'Executive-level': 3
}
df_encoded['exp_level']= df_encoded[column_name2].map(exp_mapping)
df_encoded=df_encoded.drop(columns='experience_level')

print(df_encoded.head())
df_encoded.to_csv('dataset_proceeded.csv', index=False)