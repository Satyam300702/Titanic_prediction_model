# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 07:59:19 2025

@author: HP
"""

import numpy as np
import os
import pickle
import streamlit as st

model_path = os.path.join(os.path.dirname(__file__),"titanic_survival.sav")
try:
    titanic = pickle.load(open(model_path,"rb"))
except FileNotFoundError:
    st.error("Model File not found")
    st.stop()
    
def  Titanic_prediction(input_data):
    input_data_as_np_array = np.asarray(input_data).reshape(1,-1)
    prediction = titanic.predict(input_data_as_np_array)
    prediction
    if prediction[0] == 1:
        return "✅ The passenger is likely to SURVIVE."
    else:
        return "❌ The passenger is NOT likely to SURVIVE."
        
def main():
    st.title("Titanic Survial Preidiction App")
        
    st.header("Passenger Information")
    pclass = st.selectbox("Passenger Class(1=1st, 2 = 2nd, 3 = 3rd",[1,2,3])
    sex = st.selectbox("Sex",["male","female"])
    age = st.number_input("Age")
    sibsp = st.number_input("Number of Siblings Aboard (SibSp")
    parch = st.number_input("Number of Parents/Children Aboard (Parch")
    fare = st.number_input("Fare")
    embarked = st.selectbox("Port of Embarkation",["C","Q","S"])
    
    sex = 0 if sex =="femalr" else 1
    if embarked == "Q":
        embarked_Q = 1
        embarked_S = 0
    elif embarked == "S":
        embarked_Q = 0
        embarked_S = 1
    else:  # "C" (baseline)
        embarked_Q = 0
        embarked_S = 0
    
    titanic_model = ""
    if st.button("Predict"):
        titanic_model = Titanic_prediction([pclass,sex,age,sibsp,parch,fare,embarked_Q,embarked_S])
    st.success(titanic_model)
    
if __name__ == "__main__":
    main()
        
        
        
        