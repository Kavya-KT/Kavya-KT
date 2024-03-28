import pickle
from PIL import Image
import streamlit as st

def main():
    st.title('Real Estate')
    image=Image.open('real-estate 1.jpeg')
    st.image(image,width=800)
    X2=st.text_input('X2 house age','text here')
    X3= st.text_input('X3 distance to the nearest MRT station', 'text here')
    X4= st.text_input('X4 number of convenience stores', 'text here')
    X5 = st.text_input('X5 latitude', 'text here')
    X6= st.text_input('X6 longitude', 'text here')

    f=[X2,X3,X4,X5,X6]
    model=pickle.load(open('realestate','rb'))
    scaler=pickle.load(open('scaler','rb'))

    pred=st.button('PREDICT')

    if(pred):
        prediction=model.predict(scaler.transform([f]))
        st.write(prediction)
main()
