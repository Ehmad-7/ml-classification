import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data=load_breast_cancer()

x=data.data
y=data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model=joblib.load('breast_cancer_model.pkl')

predictions=model.predict(x_test)

accuracy=accuracy_score(y_test,predictions)
print(f'Accuracy: {accuracy:.2f}')