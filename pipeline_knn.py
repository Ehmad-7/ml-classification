from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()
x=data.data
y=data.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

pipeline=Pipeline([
    ('scaler',StandardScaler()),
    ('classifier',KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(x_train,y_train)

y_pred=pipeline.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print('Accuracy',accuracy)