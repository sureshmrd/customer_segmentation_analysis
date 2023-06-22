# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 09:19:18 2023

@author: Suresh
"""
import numpy as np
import pickle
import pandas as pd
import streamlit as st


loaded_model=pickle.load(open('trained_model_cust_seg.sav','rb'))

#creating prediction
def customer_seg(input_data):
    #converting input data to numerical values
   # input_data=[int(x) for x in input_data]
    
    #changing input data to numpy array
    #input_data_as_numpy_array=np.asarray(input_data)
    
    #reshaping the array as we are predicting only for one instance
    #input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    
    prediction=loaded_model.fit_predict(input_data)
    
 
    return prediction
    
def main():
    st.title("customer segmentation")
    
    Annual_Income1=st.text_input('annual income1')
    spending_score1=st.text_input('spending score1')
    list1=[Annual_Income1,spending_score1]
    
    Annual_Income2=st.text_input('annual income2')
    spending_score2=st.text_input('spending score2')
    list2=[Annual_Income2,spending_score2]
    
    Annual_Income3=st.text_input('annual income3')
    spending_score3=st.text_input('spending score3')
    list3=[Annual_Income3,spending_score3]
    
    Annual_Income4=st.text_input('annual income4')
    spending_score4=st.text_input('spending score4')
    list4=[Annual_Income4,spending_score4]
    
    Annual_Income5=st.text_input('annual income5')
    spending_score5=st.text_input('spending score5')
    list5=[Annual_Income5,spending_score5]
    
    input_list=[list1,list2,list3,list4,list5]
    
    result=[]
    
    if st.button("user result"):
        result=customer_seg(input_list)
        
    st.success(result)
    
    
    
if __name__=='__main__':
    main()
        
        
    
    
    
    
    
    
    
    

