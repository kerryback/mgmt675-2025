
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Set page title
st.title('Tangency Portfolio Calculator')

# User inputs
num_tickers = st.number_input('Enter number of tickers:', min_value=1, value=5)
tickers = []
for i in range(int(num_tickers)):
    ticker = st.text_input(f'Enter ticker {i+1}:', value='SPY' if i==0 else '')
    if ticker:
        tickers.append(ticker)

rf_rate = st.number_input('Enter monthly risk-free rate (as decimal):', value=0.04/12, format='%f')

if st.button('Calculate Portfolio') and len(tickers) == num_tickers:
    try:
        # Download data
        start_date = '1970-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = pd.DataFrame()
        for ticker in tickers:
            temp = yf.download(ticker, start=start_date, end=end_date, interval='1mo')['Close']
            data[ticker] = temp
            
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Calculate optimal weights
        excess_returns = mean_returns - rf_rate
        inv_cov = np.linalg.inv(cov_matrix)
        weights = inv_cov.dot(excess_returns)
        weights = weights/weights.sum()
        
        # Calculate portfolio statistics
        port_return = np.sum(weights * mean_returns)
        port_vol = np.sqrt(weights.dot(cov_matrix).dot(weights))
        
        # Display results
        st.write('Tangency Portfolio Weights:')
        for ticker, weight in zip(tickers, weights):
            st.write(f'{ticker}: {weight:.2%}')
            
        st.write('\nPortfolio Statistics (Monthly):')
        st.write(f'Expected Return: {port_return:.2%}')
        st.write(f'Standard Deviation: {port_vol:.2%}')
        
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
