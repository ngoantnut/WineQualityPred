import pickle
import warnings

import numpy as np
import streamlit as st
import pandas as pd
warnings.filterwarnings("ignore")

st.image('wine.jpg')
st.title('Dự đoán chất lượng rượu vang: ')
col1, col2, col3 =st.columns(3)

# Id= np.str(col1.text_input("Id"))
fixed_acidity= np.str(col1.text_input('fixed acidity'))
volatile_acidity= np.str(col2.text_input('volatile acidity'))
citric_acid= np.str(col3.text_input('citric acid'))
residual_sugar= np.str(col1.text_input('residual sugar'))
chlorides= np.str(col2.text_input('chlorides'))
free_sulfur_dioxide= np.str(col3.text_input('free sulfur dioxide'))
total_sulfur_dioxide= np.str(col1.text_input('total sulfur dioxide'))
density= np.str(col2.text_input('density'))
pH= np.str(col3.text_input('pH'))
sulphates= np.str(col1.text_input('sulphates'))
alcohol= np.str(col2.text_input('alcohol'))

features_num = ['fixed acidity','volatile acidity', 'citric acid',
                'residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide',
                'density','pH','sulphates', 'alcohol']
result= st.button('Dự đoán ')
sample=[ fixed_acidity,volatile_acidity, citric_acid,
         residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide, density,pH,sulphates, alcohol]
sample_df=pd.DataFrame([sample], columns=features_num)

model = pickle.load(open("https://github.com/ngoantnut/WineQualityPred/blob/main/finalized_model.sav","rb"))
if result:
    st.write('Chất lượng của rượu vang là:  ' , str(model.predict(sample_df)))
if result >= 8:
    st.write('Rượu chất lượng tốt!')
if result >=6:
    st.write('Rượu khá')
else:
    st.write('Rượu tệ!')
st.info('Bài tập lớn Trí tuệ nhân tạo - Trần Ngoan')
st.info('Mssv: K185480106014 ')
st.info(' Email:tvngoan343@gmail.com')
