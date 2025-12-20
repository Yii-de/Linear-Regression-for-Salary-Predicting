import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('dataset_proceeded.csv')

X = df.drop(columns='salary_in_usd')
y = df['salary_in_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

model = LinearRegression()
model.fit(X_train, y_train)

def regFormula(model, feature_names, target_name='salary_in_usd'):
    b=model.intercept_
    w=model.coef_
    equation = f"{target_name} = {b:.5f}"
    for coef, name in zip(w, feature_names):
        sign = " +" if coef >= 0 else " -"
        equation += f"{sign} {abs(coef):.5f}*{name}" 
    print(equation)

y_pred = model.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
print (f"Mean Square Error: {mse:.4f}")
print (f"R-Squared: {r2:.4f}")

feature_names = [
    'emp_is_UnitedStates',
    'work_model_On-site',
    'work_model_Remote',
    'comp_is_US',
    'company_size_encoded',
    'job_title_group',
    'exp_level'
]
regFormula(model, feature_names)