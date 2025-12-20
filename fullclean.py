import pandas as pd
import numpy as np

df = pd.read_csv('data_science_salaries.csv')

df['emp_is_UnitedStates'] = np.where(df['employee_residence']=='United States',1,0)
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

top_3_jobs = df_encoded['job_title'].value_counts().head(3).index.tolist()
df_encoded['job_title_group'] = df_encoded['job_title'].isin(top_3_jobs).astype(int)
df_encoded=df_encoded.drop(columns='job_title')

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