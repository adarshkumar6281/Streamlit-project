import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris=load_iris()
x,y= iris.data,iris.target
model=RandomForestClassifier(n_estimators=50,random_state=42)
model.fit(x,y)
joblib.dump(model,"model.pkl")
print("model trained and exported successfully")



