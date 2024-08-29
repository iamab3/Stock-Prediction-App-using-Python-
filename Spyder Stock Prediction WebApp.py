#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:04:41 2024

@author: abhishekjain
"""

import streamlit as st
from datetime import date

import yfinance as yf
import prophet as Prophet
import prophet.plot as plot_plotly
from plotly import graph_objs as go

start = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "MIN.AX")
selected_stock = st.selectbox("Select Company Ticker for prediction", stocks)

n_years = st.slider("Years of prediction: ", 1, 4)

period = n_years*365

# To cache the data to avoid downloading again and again
@st.cache_data

def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader("Raw Data")
st.write(data.tail())

# Plotting the raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name = 'Stock_Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name = 'Stock_Close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

m = Prophet.Prophet(weekly_seasonality= True)
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plotting the forecasted data
st.subheader('Forecast Data')
st.write(forecast.tail())

st.write("Forecast Data")
fig1 = m.plot(forecast)
st.plotly_chart(fig1)

st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)










































