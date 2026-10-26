import joblib
import numpy as np
import streamlit as st
st.set_page_config(page_title="iris flower classification",layout="centered")
st.title("iris flower classification")
st.write("adjust the feature sliders below to predict the target species")
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
model= load_model()
st.sidebar.header("input features")
sepal_len = st.sidebar.slider("sepallength(cm)",4.0,8.0,5.8)
sepal_wid = st.sidebar.slider("sepalwidth(cm)",2.0,4.5,3.0)
petal_len = st.sidebar.slider("petallength(cm)",1.0,7.0,4.35)
petal_wid = st.sidebar.slider("petalwidth(cm)",0.1,2.5,1.3)
target_names = ["Setosa","Versicolor","Virginica"]
if st.button("predict species"):
    features=np.array([[sepal_len,sepal_wid,petal_len,petal_wid]])
    prediction= model.predict(features)[0]
    probabilities= model.predict_proba(features)[0]
    st.success(f"predicted species:iris-{target_names[prediction]}")
    st.subheader("confidence score")
    for name,prob in zip(target_names,probabilities):
        st.write(f"{name}:{prob*100:.1f}%")
