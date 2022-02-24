# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 22:28:18 2022

@author: sarmi
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('D:/ML/trained_model.sav', 'rb'))

# creating a function for prediction

def diabetes_prediction(input_data):
   

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    #giving a title
    st.title('Diabetes Predictor')
    
    #Getting input 
    Pregnancies = st.text_input("Number of pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("BP level")
    SkinThickness = st.text_input(" Skin thickness ") 
    Insulin = st.text_input("Insulin level") 
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes pedigree function value")
    Age = st.text_input("Age of the person") 
    
#Prediction code
    diagnosis=''
    
    #Button for prediction
    if st.button("Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
