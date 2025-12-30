import pandas as pd
import numpy as np

df = pd.read_csv('data_science_salaries.csv')

df_encoded = pd.get_dummies(df, columns=['work_models'], drop_first=True, dtype=int)
df_encoded = df_encoded.drop(columns='employee_residence')

df_encoded['comp_is_US'] = np.where(df_encoded['company_location']=='United States', 1, 0)
df_encoded = df_encoded.drop(columns='company_location')

size_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
df_encoded['company_size_encoded'] = df_encoded['company_size'].map(size_mapping)
df_encoded = df_encoded.drop(columns='company_size')

df_encoded['salary_in_usd'] = np.log(df_encoded['salary_in_usd']).round(3)

Q1 = df_encoded['salary_in_usd'].quantile(0.25)
Q3 = df_encoded['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_encoded = df_encoded[(df_encoded['salary_in_usd'] >= lower_bound) & 
                        (df_encoded['salary_in_usd'] <= upper_bound)]

def jobClassify(title):
    title = str(title).lower()
    if any(k in title for k in ['scientist', 'science', 'research', 'nlp', 'deep learning']):
        return 'Data Science'
    if any(k in title for k in ['ml', 'machine learning', 'ai', 'mlops', 'computer vision']):
        return 'Machine Learning / AI'
    if any(k in title for k in ['engineer', 'infrastructure', 'etl', 'architect', 'developer']):
        return 'Data Engineering'
    if any(k in title for k in ['analyst', 'bi', 'analytics', 'insights', 'visualization']):
        return 'Data Analysis'
    return 'Other'

df_encoded['domain_expertise'] = df_encoded['job_title'].apply(jobClassify)
df_encoded = df_encoded.drop(columns='job_title')
df_encoded = pd.get_dummies(df_encoded, columns=['domain_expertise'], drop_first=True, dtype=int)

exp_mapping = {
    'Entry-level': 0,
    'Senior-level': 1,
    'Mid-level': 2,
    'Executive-level': 3
}
df_encoded['exp_level'] = df_encoded['experience_level'].map(exp_mapping)
df_encoded = df_encoded.drop(columns='experience_level')

df_encoded.to_csv('dataset_proceeded.csv', index=False)

print("Đã lưu dữ liệu sạch vào file dataset_proceeded.csv")
