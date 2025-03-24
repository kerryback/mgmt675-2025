
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
import pandas as pd

st.title('Portfolio Optimization App')

# Input number of assets
n_assets = st.number_input('Enter number of risky assets:', min_value=2, value=3, step=1)

# Input asset names
asset_names = []
for i in range(n_assets):
    name = st.text_input(f'Asset {i+1} name:', value=f'Asset {i+1}')
    asset_names.append(name)

# Input expected returns
st.subheader('Expected Returns')
returns = []
for i in range(n_assets):
    ret = st.number_input(f'Expected return for {asset_names[i]}:', value=0.10, format='%.4f')
    returns.append(ret)
returns = np.array(returns)

# Input covariance matrix
st.subheader('Covariance Matrix')
cov_matrix = np.zeros((n_assets, n_assets))
for i in range(n_assets):
    for j in range(i, n_assets):
        if i == j:
            val = st.number_input(f'Variance of {asset_names[i]}:', value=0.04, format='%.4f')
        else:
            val = st.number_input(f'Covariance between {asset_names[i]} and {asset_names[j]}:', value=0.02, format='%.4f')
        cov_matrix[i,j] = val
        cov_matrix[j,i] = val

# Input risk-free rate
rf = st.number_input('Risk-free rate:', value=0.02, format='%.4f')

def portfolio_stats(weights, returns, cov_matrix):
    portfolio_return = np.sum(returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std

def negative_sharpe_ratio(weights, returns, cov_matrix, rf):
    p_ret, p_std = portfolio_stats(weights, returns, cov_matrix)
    return -(p_ret - rf) / p_std

def optimize_portfolio(returns, cov_matrix, target_return=None):
    n = len(returns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(returns * x) - target_return})
    bounds = tuple((0, 1) for _ in range(n))
    
    if target_return is None:
        # Minimize volatility
        result = minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
                         np.ones(n)/n,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
    else:
        # Minimize volatility for given return
        result = minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
                         np.ones(n)/n,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
    
    return result.x

if st.button('Calculate Optimal Portfolios'):
    # Calculate tangency portfolio
    init_weights = np.ones(n_assets) / n_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    result = minimize(negative_sharpe_ratio, init_weights,
                     args=(returns, cov_matrix, rf),
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    tangency_weights = result.x
    tangency_return, tangency_std = portfolio_stats(tangency_weights, returns, cov_matrix)
    
    # Calculate efficient frontier
    target_returns = np.linspace(min(returns), max(returns), 100)
    efficient_returns = []
    efficient_stds = []
    portfolio_weights = []
    
    for target in target_returns:
        try:
            weights = optimize_portfolio(returns, cov_matrix, target)
            ret, std = portfolio_stats(weights, returns, cov_matrix)
            efficient_returns.append(ret)
            efficient_stds.append(std)
            portfolio_weights.append(weights)
        except:
            continue
    
    # Create interactive plot
    fig = go.Figure()
    
    # Plot individual assets
    fig.add_trace(go.Scatter(
        x=[np.sqrt(cov_matrix[i,i]) for i in range(n_assets)],
        y=returns,
        mode='markers',
        name='Individual Assets',
        text=asset_names,
        hovertemplate='Asset: %{text}<br>Return: %{y:.4f}<br>Std Dev: %{x:.4f}'
    ))
    
    # Plot efficient frontier
    hover_text = []
    for weights in portfolio_weights:
        text = '<br>'.join([f'{asset}: {w:.4f}' for asset, w in zip(asset_names, weights)])
        hover_text.append(text)
    
    fig.add_trace(go.Scatter(
        x=efficient_stds,
        y=efficient_returns,
        mode='lines',
        name='Efficient Frontier',
        text=hover_text,
        hovertemplate='Return: %{y:.4f}<br>Std Dev: %{x:.4f}<br>Weights:<br>%{text}'
    ))
    
    # Plot tangency portfolio
    fig.add_trace(go.Scatter(
        x=[tangency_std],
        y=[tangency_return],
        mode='markers',
        name='Tangency Portfolio',
        marker=dict(size=10),
        text=['<br>'.join([f'{asset}: {w:.4f}' for asset, w in zip(asset_names, tangency_weights)])],
        hovertemplate='Return: %{y:.4f}<br>Std Dev: %{x:.4f}<br>Weights:<br>%{text}'
    ))
    
    # Plot Capital Allocation Line
    cal_x = np.linspace(0, max(efficient_stds)*1.2, 100)
    cal_y = rf + (tangency_return - rf) * cal_x / tangency_std
    
    fig.add_trace(go.Scatter(
        x=cal_x,
        y=cal_y,
        mode='lines',
        name='Capital Allocation Line',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='Portfolio Optimization Results',
        xaxis_title='Standard Deviation',
        yaxis_title='Expected Return',
        hovermode='closest'
    )
    
    st.plotly_chart(fig)
    
    # Display tangency portfolio weights
    st.subheader('Tangency Portfolio Weights:')
    weights_df = pd.DataFrame({
        'Asset': asset_names,
        'Weight': tangency_weights
    })
    st.dataframe(weights_df)
    
    # Display portfolio statistics
    st.subheader('Portfolio Statistics:')
    st.write(f'Tangency Portfolio Return: {tangency_return:.4f}')
    st.write(f'Tangency Portfolio Standard Deviation: {tangency_std:.4f}')
    st.write(f'Sharpe Ratio: {(tangency_return - rf)/tangency_std:.4f}')
