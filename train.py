from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

data=load_breast_cancer(as_frame=True)
df=data.frame

print(df.sample(3))
print(df.info())

x=df.drop(columns=['target'])
y=df['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

joblib.dump(model,'breast_cancer_model.pkl')

print('Logistic RegressionModel saved as breast_cancer_model.pkl')