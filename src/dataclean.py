import pandas as pd
import numpy as np

df = pd.read_csv('data_science_salaries.csv')
df.info()
df.head()
print(df.duplicated())

df['is_UnitedStates'] = np.where(df['employee_residence']=='United States',1,0)
df_encoded = pd.get_dummies(df , columns=['work_models'], drop_first=True, dtype=int)
print(df_encoded)
df_encoded=df_encoded.drop(columns='employee_residence')
print(df_encoded.head())
df_encoded['comp_is_US'] = np.where(df['company_location']=='United States',1,0)
print(df_encoded)
df_encoded=df_encoded.drop(columns='company_location')
print(df_encoded.head())
column_name = 'company_size'
size_mapping = {
    'Small': 0,
    'Medium': 1,
    'Large': 2
}
df['company_size_encoded'] = df[column_name].map(size_mapping)
#df_encoded.to_csv('new_dataset.csv', index=False)