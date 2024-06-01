import streamlit as st
import pickle
from PIL import Image
def main():
    st.title(':blue[WATER ] POTABILITY:droplet:')
    image=Image.open('water image.jpeg')
    st.image(image,width=500)
    on = st.toggle('Features')
    if on:
        a=st.number_input('ph',min_value=0.00,max_value=14.00)
        # a=st.text_input('ph','type here')
        b=st.text_input('hardness','text here')
        c=st.text_input('solid','text here')
        d=st.text_input('chloramines','text here')
        e=st.text_input('sulfate','text here')
        f=st.text_input('conductivity','text here')
        g=st.text_input('organic_carbon','text here')
        h=st.text_input('trihalomethanes','text here')
        i=st.text_input('turbidity','text here')
        fe=[a,b,c,d,e,f,g,h,i]
        model=pickle.load(open('water1','rb'))
        scaler=pickle.load((open('scaler1','rb')))
        pred=st.button(':rainbow[PREDICT]')
        if pred:
            prediction=model.predict(scaler.transform([fe]))
            if prediction==0:
                st.write('non potable ')
                with st.expander("See explanation"):
                    st.write('Non-potable water can contain chemicals from industry and agriculture, human or animal waste, water treatment and distribution, or natural contaminants (when water travels through soil).'
                             ' Soil can contain arsenic, heavy metals and pesticide residues.')
            else:
                st.write('potable')
                with st.expander("See explanation"):
                    st.write('Potable water is the water which is filtered and treated properly and is finally free from all contaminants and harmful bacteria. This purified water is fit to drink, or it can be called '
                             'drinking water after the purification processes and is safe for both cooking and drinking.')
main()