##install streamlit
#pip list-->to get all pip elemnts
#pip install streamlit

import streamlit as st
import pickle
from PIL import Image
from PIL.IcoImagePlugin import IcoImageFile


#create a function
def main():
    #to add titile
    st.title(':red[HEART FAILURE PREDICTION]')
    #to read image
    image=Image.open('heart_failure_image.jpeg')
    st.image(image,width=800)

    #identitfy the features

    #input features
    age=st.text_input('Age','Type here')
    #to add radio button
    sex=st.radio('sex',['Male','Female'])
    if(sex=='Male'):
        sex=1
    else:
        sex=0
    cp= st.text_input('cp', 'Type here')
    trestbps = st.text_input('trestbps', 'Type here')
    chol = st.text_input('chol', 'Type here')
    fbs = st.text_input('fbs', 'Type here')
    restecg = st.text_input('restecg', 'Type here')
    thalach = st.text_input('thalach', 'Type here')
    exang = st.text_input('exang', 'Type here')
    oldpeak = st.text_input('oldpeak', 'Type here')
    slope = st.text_input('slope', 'Type here')
    ca = st.text_input('ca', 'Type here')
    thal = st.text_input('thal', 'Type here')


    #store all feartures in a vraiable

    f=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

    #load the stored model and scaler
    model1=pickle.load(open('model_knn.sav','rb'))
    scaler1=pickle.load(open('scaler_knn.save','rb'))

    #to predict we add a button
    pred=st.button('PREDICT')

    #enable button
    if pred:
        prediction=model1.predict(scaler1.transform([f]))#single score bracket bcz features alresdy in list format
        if prediction==0:
        #to print use write
            st.write('not suffering heart diseaese')
        else:
            st.write('suffering heart disease')

main()
