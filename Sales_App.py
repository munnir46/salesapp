import joblib
import pandas as pd
import streamlit as st

model = joblib.load('../models/sales_model.pkl')
encoder = joblib.load("../models/sales_encoder.pkl")

#Accept user input
month = st.selectbox ('Select month', ['January', 'February', 'March', 'April', 'May', 'June', 'july', 'August', 'September', 'October', 'November', 'December' ])
product = st.text_input("Enter product (eg. toys):")
promotion = st.selectbox("Was there promotion(0 or 1)?:",[0, 1])
holiday = st.selectbox("Was there holiday(0 or 1)?:", [0, 1])


if st.button('Predict sales'):
    predict_data = pd.DataFrame({
      "month": [month],
      "product":[product],
      "promotion": [promotion],
      "holiday": [holiday]
     })

#turn them to lowercase
    predict_data['month'] = predict_data['month'].str.lower()
    predict_data['product'] = predict_data['product'].str.lower()

# Transform the predict_data into encoded
    predictdata = encoder.transform(predict_data)

    result = model.predict(predictdata)

    st.success(f"Prediction: {result}")
