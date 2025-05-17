
import streamlit as st
import yfinance as yf
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize

# Set page configuration
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Stock Market Dashboard")
st.markdown("### A simple dashboard to analyze stock market data and optimize portfolios")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Number of stocks to analyze
num_stocks = st.sidebar.slider("Number of stocks to analyze", min_value=2, max_value=10, value=3)

# Create input fields for each stock
stock_symbols = []
for i in range(num_stocks):
    default_symbol = ""
    if i == 0:
        default_symbol = "AAPL"
    elif i == 1:
        default_symbol = "MSFT"
    elif i == 2:
        default_symbol = "GOOG"
    
    symbol = st.sidebar.text_input(f"Stock Symbol {i+1}", default_symbol)
    stock_symbols.append(symbol)

# Date range selection
today = datetime.today()
default_start = today - timedelta(days=365*5)  # 5 years of data for better analysis
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", today)

# Risk-free rate for CAPM and portfolio optimization
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100

# Fetch data button
if st.sidebar.button("Fetch Data"):
    # Remove empty symbols
    stock_symbols = [symbol for symbol in stock_symbols if symbol]
    
    if len(stock_symbols) < 2:
        st.error("Please enter at least two stock symbols for portfolio analysis.")
    else:
        # Fetch price data
        @st.cache_data
        def load_price_data(tickers, start, end):
            data = {}
            for ticker in tickers:
                try:
                    stock_data = yf.download(ticker, start=start, end=end)
                    if not stock_data.empty:
                        data[ticker] = stock_data
                except Exception as e:
                    st.error(f"Error fetching price data for {ticker}: {e}")
            return data
        
        # Load the data
        with st.spinner('Fetching price data...'):
            price_data = load_price_data(stock_symbols, start_date, end_date)
        
        if price_data and len(price_data) >= 2:
            # Create tabs for different analysis views
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Comparison", "Individual Analysis", "Correlation", "Expected Returns", "Portfolio Optimization"])
            
            with tab1:
                st.subheader("Stock Price Comparison")
                
                # Create a dataframe with closing prices for all stocks
                close_prices = pd.DataFrame()
                for symbol, data in price_data.items():
                    close_prices[symbol] = data['Close']
                
                # Display the dataframe
                st.write("Closing Prices")
                st.write(close_prices.head())
                
                # Plot all stocks on the same chart
                fig = go.Figure()
                for symbol in close_prices.columns:
                    fig.add_trace(go.Scatter(x=close_prices.index, y=close_prices[symbol], name=symbol))
                
                fig.update_layout(title="Stock Price Comparison", xaxis_title="Date", yaxis_title="Price (USD)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Normalized prices (percentage change)
                st.subheader("Normalized Price Comparison (Base = 100)")
                normalized = pd.DataFrame()
                for symbol in close_prices.columns:
                    normalized[symbol] = close_prices[symbol] / close_prices[symbol].iloc[0] * 100
                
                st.write("Normalized Prices (First day = 100)")
                st.write(normalized.head())
                
                # Plot normalized prices
                fig2 = go.Figure()
                for symbol in normalized.columns:
                    fig2.add_trace(go.Scatter(x=normalized.index, y=normalized[symbol], name=symbol))
                
                fig2.update_layout(title="Normalized Price Comparison", xaxis_title="Date", yaxis_title="Normalized Price (Base=100)")
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab2:
                st.subheader("Individual Stock Analysis")
                
                # Select stock for individual analysis
                selected_stock = st.selectbox("Select a stock for detailed analysis", list(price_data.keys()))
                
                if selected_stock:
                    data = price_data[selected_stock]
                    
                    # Display basic info
                    st.write(f"Data for {selected_stock}")
                    st.write(data.head())
                    
                    # Display statistics
                    st.write("Summary Statistics")
                    st.write(data.describe())
                    
                    
                    # Plot stock price
                    st.subheader("Stock Price Chart")
                    fig = go.Figure()
                    # Ensure we're using all available data points
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
                    fig.update_layout(
                        title=f"{selected_stock} Stock Price", 
                        xaxis_title="Date", 
                        yaxis_title="Price (USD)",
                        xaxis=dict(
                            rangeslider=dict(visible=True),
                            type="date"
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    # Calculate and plot moving averages
                    st.subheader("Moving Averages")
                    data['MA20'] = data['Close'].rolling(window=20).mean()
                    data['MA50'] = data['Close'].rolling(window=50).mean()
                    data['MA200'] = data['Close'].rolling(window=200).mean()
                    
                    
                    fig2 = go.Figure()
                    # Ensure we're using all available data points
                    fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
                    fig2.add_trace(go.Scatter(x=data.index, y=data['MA20'], name="20-day MA"))
                    fig2.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="50-day MA"))
                    fig2.add_trace(go.Scatter(x=data.index, y=data['MA200'], name="200-day MA"))
                    fig2.update_layout(
                        title=f"{selected_stock} Moving Averages", 
                        xaxis_title="Date", 
                        yaxis_title="Price (USD)",
                        xaxis=dict(
                            rangeslider=dict(visible=True),
                            type="date"
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                    # Volume chart
                    st.subheader("Trading Volume")
                    fig3 = go.Figure()
                    # Ensure we're using all available data points
                    fig3.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume"))
                    fig3.update_layout(
                        title=f"{selected_stock} Trading Volume", 
                        xaxis_title="Date", 
                        yaxis_title="Volume",
                        xaxis=dict(
                            rangeslider=dict(visible=True),
                            type="date"
                        )
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    with tab3:
                        st.subheader("Correlation Analysis")
                        # Create correlation matrix
                        correlation_df = pd.DataFrame()
                        for symbol, data in price_data.items():
                            correlation_df[symbol] = data['Close']
                        correlation_matrix = correlation_df.corr()
                        # Display correlation matrix
                        st.dataframe(correlation_matrix)
                        fig, ax = plt.subplots()
                        cax = ax.matshow(correlation_matrix, cmap="coolwarm")
                        fig.colorbar(cax)
                        ax.set_xticks(range(len(correlation_matrix.columns)))
                        ax.set_yticks(range(len(correlation_matrix.index)))
                        ax.set_xticklabels(correlation_matrix.columns, rotation=90)
                        ax.set_yticklabels(correlation_matrix.index)
                        st.pyplot(fig)
                
                # Create correlation matrix
                correlation_df = pd.DataFrame()
                for symbol, data in price_data.items():
                    correlation_df[symbol] = data['Close']
                
                correlation_matrix = correlation_df.corr()
                
                # Display correlation matrix
                st.write("Correlation Matrix")
                st.write(correlation_matrix)
                
                # Heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='Viridis',
                    zmin=-1, zmax=1
                ))
                fig.update_layout(title="Correlation Heatmap", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate daily returns
                st.subheader("Daily Returns")
                returns_df = pd.DataFrame()
                for symbol, data in price_data.items():
                    returns_df[symbol] = data['Close'].pct_change() * 100
                
                # Remove first row (NaN)
                returns_df = returns_df.dropna()
                
                # Display returns
                st.write("Daily Returns (%)")
                st.write(returns_df.head())
                
                # Plot returns
                fig2 = go.Figure()
                for symbol in returns_df.columns:
                    fig2.add_trace(go.Scatter(x=returns_df.index, y=returns_df[symbol], name=symbol, mode='lines'))
                
                fig2.update_layout(title="Daily Returns Comparison", xaxis_title="Date", yaxis_title="Daily Return (%)")
                st.plotly_chart(fig2, use_container_width=True)
            
            with tab4:
                st.subheader("Expected Returns Analysis")
                
                # Calculate daily returns for all stocks
                returns_df = pd.DataFrame()
                for symbol, data in price_data.items():
                    returns_df[symbol] = data['Close'].pct_change()
                
                # Remove first row (NaN)
                returns_df = returns_df.dropna()
                
                # Calculate annualized returns and volatility
                mean_daily_returns = returns_df.mean()
                annual_returns = (1 + mean_daily_returns) ** 252 - 1  # 252 trading days in a year
                
                daily_std = returns_df.std()
                annual_std = daily_std * np.sqrt(252)  # Annualized volatility
                
                # Calculate Sharpe Ratio
                sharpe_ratios = (annual_returns - risk_free_rate) / annual_std
                
                # Create a summary dataframe
                summary_df = pd.DataFrame({
                    'Annual Return (%)': annual_returns * 100,
                    'Annual Volatility (%)': annual_std * 100,
                    'Sharpe Ratio': sharpe_ratios
                })
                
                # Sort by Sharpe Ratio
                summary_df = summary_df.sort_values('Sharpe Ratio', ascending=False)
                
                st.write("Expected Returns and Risk Metrics")
                st.write(summary_df)
                
                # Plot risk vs return
                fig = px.scatter(
                    summary_df, 
                    x='Annual Volatility (%)', 
                    y='Annual Return (%)',
                    text=summary_df.index,
                    size='Sharpe Ratio',
                    color='Sharpe Ratio',
                    color_continuous_scale='Viridis',
                    title="Risk vs Return"
                )
                
                fig.update_traces(textposition='top center')
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate Beta for each stock (using S&P 500 as market)
                st.subheader("Beta Analysis")
                
                try:
                    # Download S&P 500 data
                    sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close']
                    sp500_returns = sp500.pct_change().dropna()
                    
                    # Align market returns with stock returns
                    aligned_returns = pd.DataFrame(sp500_returns)
                    aligned_returns.columns = ['Market']
                    
                    for symbol in returns_df.columns:
                        # Use statsmodels for regression (more robust than manual calculation)
                        X = aligned_returns['Market'].values.reshape(-1, 1)
                        y = aligned_returns[symbol].values
                        
                        # Add constant for regression
                        X = sm.add_constant(X)
                        
                        # Fit regression model
                        model = sm.OLS(y, X).fit()
                        
                        # Beta is the slope coefficient
                        beta = model.params[1]
                        betas[symbol] = beta
                    
                    # Create Beta dataframe
                    beta_df = pd.DataFrame(list(betas.items()), columns=['Stock', 'Beta'])
                    beta_df = beta_df.sort_values('Beta', ascending=False)
                    
                    st.write("Beta Values (Relative to S&P 500)")
                    st.write(beta_df)
                    
                    # Plot Beta values
                    fig2 = px.bar(
                        beta_df,
                        x='Stock',
                        y='Beta',
                        color='Beta',
                        color_continuous_scale='RdBu',
                        title="Beta Values"
                    )
                    
                    fig2.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="Market Beta = 1")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Calculate expected returns using CAPM
                    market_return = (1 + sp500_returns.mean()) ** 252 - 1  # Annualized market return
                    
                    capm_returns = {}
                    for symbol, beta in betas.items():
                        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
                        capm_returns[symbol] = expected_return
                    
                    # Create CAPM dataframe
                    capm_df = pd.DataFrame(list(capm_returns.items()), columns=['Stock', 'CAPM Expected Return (%)'])
                    capm_df['CAPM Expected Return (%)'] = capm_df['CAPM Expected Return (%)'] * 100
                    capm_df = capm_df.sort_values('CAPM Expected Return (%)', ascending=False)
                    
                    st.write("CAPM Expected Returns")
                    st.write(capm_df)
                    
                    # Plot CAPM expected returns
                    fig3 = px.bar(
                        capm_df,
                        x='Stock',
                        y='CAPM Expected Return (%)',
                        color='CAPM Expected Return (%)',
                        color_continuous_scale='Viridis',
                        title="CAPM Expected Returns"
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error calculating Beta: {e}")
            
            with tab5:
                st.subheader("Portfolio Optimization")
                # Portfolio weight constraints
                st.sidebar.header("Portfolio Constraints")
                
                # Add option to toggle shorting
                allow_shorting = st.sidebar.checkbox("Allow Shorting Positions", value=True)
                
                # Add sliders for min and max weight constraints
                min_weight = st.sidebar.slider("Minimum Weight (%)", min_value=-30, max_value=0, value=-3, step=1) / 100
                max_weight = st.sidebar.slider("Maximum Weight (%)", min_value=10, max_value=100, value=50, step=5) / 100
                
                if not allow_shorting and min_weight < 0:
                    min_weight = 0
                    st.sidebar.warning("Minimum weight set to 0% since shorting is disabled")
                
                st.write(f"Portfolio Constraints: Min Weight = {min_weight*100:.1f}%, Max Weight = {max_weight*100:.1f}%")
                if allow_shorting:
                    st.write("Shorting is enabled: Negative weights are allowed")
                else:
                    st.write("Shorting is disabled: Only positive weights are allowed")
                
                
                st.markdown('''
                ### Portfolio Constraints
                
                This optimization allows for:
                - **Long positions**: Buying stocks (positive weights)
                - **Short positions**: Selling borrowed stocks (negative weights)
                - **Weight constraints**: Limiting exposure to any single stock
                
                Shorting can potentially increase returns but also increases risk.
                ''')
                
                # Calculate daily returns for all stocks
                returns_df = pd.DataFrame()
                for symbol, data in price_data.items():
                    returns_df[symbol] = data['Close'].pct_change()
                
                # Remove first row (NaN)
                returns_df = returns_df.dropna()
                
                # Calculate mean returns and covariance matrix
                mean_returns = returns_df.mean() * 252  # Annualized returns
                cov_matrix = returns_df.cov() * 252  # Annualized covariance
                
                # Display covariance matrix
                st.write("Annualized Covariance Matrix")
                st.write(cov_matrix)
                
                # Number of assets
                num_assets = len(returns_df.columns)
                
                # Portfolio optimization functions
                def portfolio_return(weights, mean_returns):
                    return np.sum(mean_returns * weights)
                
                def portfolio_volatility(weights, cov_matrix):
                    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
                    p_return = portfolio_return(weights, mean_returns)
                    p_volatility = portfolio_volatility(weights, cov_matrix)
                    return -(p_return - risk_free_rate) / p_volatility
                
                # Constraints and bounds
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                
                # Set bounds based on user-defined constraints
                if allow_shorting:
                    bounds = tuple((min_weight, max_weight) for asset in range(num_assets))
                else:
                    bounds = tuple((0, max_weight) for asset in range(num_assets))
                
                # Initial guess (equal weights)
                initial_guess = np.array([1/num_assets] * num_assets)
                
                # Optimize for maximum Sharpe ratio
                optimal_sharpe = minimize(
                    negative_sharpe_ratio, 
                    initial_guess, 
                    args=(mean_returns, cov_matrix, risk_free_rate), 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints
                )
                
                optimal_weights = optimal_sharpe['x']
                
                # Calculate optimal portfolio metrics
                optimal_return = portfolio_return(optimal_weights, mean_returns)
                optimal_volatility = portfolio_volatility(optimal_weights, cov_matrix)
                optimal_sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility
                
                # Display optimal portfolio
                st.subheader("Optimal Portfolio (Maximum Sharpe Ratio)")
                
                # Create a dataframe with optimal weights
                optimal_portfolio = pd.DataFrame({
                    'Stock': returns_df.columns,
                    'Weight (%)': [weight * 100 for weight in optimal_weights]
                })
                
                optimal_portfolio = optimal_portfolio.sort_values('Weight (%)', ascending=False)
                
                st.write(optimal_portfolio)
                
                # Display optimal portfolio metrics
                st.write(f"Expected Annual Return: {optimal_return*100:.2f}%")
                st.write(f"Expected Annual Volatility: {optimal_volatility*100:.2f}%")
                st.write(f"Sharpe Ratio: {optimal_sharpe_ratio:.2f}")
                
                # Calculate and display long/short exposure
                long_exposure = sum([w for w in optimal_weights if w > 0])
                short_exposure = sum([w for w in optimal_weights if w < 0])
                net_exposure = long_exposure + short_exposure  # Should be 1.0
                gross_exposure = long_exposure - short_exposure
                
                st.write("Portfolio Exposure:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Long Exposure: {long_exposure*100:.2f}%")
                    st.write(f"Short Exposure: {short_exposure*100:.2f}%")
                with col2:
                    st.write(f"Net Exposure: {net_exposure*100:.2f}%")
                    st.write(f"Gross Exposure: {gross_exposure*100:.2f}%")

                
                # Calculate and display long/short exposure
                long_exposure = sum([w for w in optimal_weights if w > 0])
                short_exposure = sum([w for w in optimal_weights if w < 0])
                net_exposure = long_exposure + short_exposure  # Should be 1.0
                gross_exposure = long_exposure - short_exposure
                
                st.write("Portfolio Exposure:")
                st.write(f"Long Exposure: {long_exposure*100:.2f}%")
                st.write(f"Short Exposure: {short_exposure*100:.2f}%")
                st.write(f"Net Exposure: {net_exposure*100:.2f}%")
                st.write(f"Gross Exposure: {gross_exposure*100:.2f}%")
                
                # Plot optimal portfolio weights
                fig = px.pie(
                    optimal_portfolio,
                    values='Weight (%)',
                    names='Stock',
                    title="Optimal Portfolio Allocation",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate the efficient frontier
                st.subheader("Efficient Frontier")
                
                # Target returns for frontier
                target_returns = np.linspace(min(mean_returns), max(mean_returns), 50)
                
                # Function to minimize portfolio variance given a target return
                def minimize_volatility(weights, cov_matrix):
                    return portfolio_volatility(weights, cov_matrix)
                
                # Function to get minimum volatility portfolio for a given target return
                def efficient_frontier_point(target_return, mean_returns, cov_matrix):
                    constraints = (
                        {'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - target_return},
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
                    )
                    
                    result = minimize(
                        minimize_volatility, 
                        initial_guess, 
                        args=(cov_matrix,), 
                        method='SLSQP', 
                        bounds=bounds, 
                        constraints=constraints
                    )
                    
                    return result['fun']
                
                # Calculate efficient frontier
                efficient_volatilities = []
                for target in target_returns:
                    efficient_volatilities.append(efficient_frontier_point(target, mean_returns, cov_matrix))
                
                # Create efficient frontier dataframe
                frontier_df = pd.DataFrame({
                    'Return (%)': target_returns * 100,
                    'Volatility (%)': [vol * 100 for vol in efficient_volatilities]
                })
                
                # Calculate individual stock metrics for plotting
                stock_returns = mean_returns * 100
                stock_volatilities = np.sqrt(np.diag(cov_matrix)) * 100
                
                # Plot efficient frontier
                fig2 = go.Figure()
                
                # Add efficient frontier
                fig2.add_trace(go.Scatter(
                    x=frontier_df['Volatility (%)'],
                    y=frontier_df['Return (%)'],
                    mode='lines',
                    name='Efficient Frontier',
                    line=dict(color='blue', width=2)
                ))
                
                # Add individual stocks
                for i, symbol in enumerate(returns_df.columns):
                    fig2.add_trace(go.Scatter(
                        x=[stock_volatilities[i]],
                        y=[stock_returns[i]],
                        mode='markers',
                        name=symbol,
                        marker=dict(size=10)
                    ))
                
                # Add optimal portfolio
                fig2.add_trace(go.Scatter(
                    x=[optimal_volatility * 100],
                    y=[optimal_return * 100],
                    mode='markers',
                    name='Optimal Portfolio',
                    marker=dict(color='red', size=15, symbol='star')
                ))
                
                fig2.update_layout(
                    title="Efficient Frontier",
                    xaxis_title="Expected Volatility (%)",
                    yaxis_title="Expected Return (%)",
                    height=600
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Monte Carlo simulation for portfolio optimization
                st.subheader("Monte Carlo Simulation")
                
                # Number of portfolios to simulate
                num_portfolios = 5000
                
                # Arrays to store results
                all_weights = np.zeros((num_portfolios, num_assets))
                ret_arr = np.zeros(num_portfolios)
                vol_arr = np.zeros(num_portfolios)
                sharpe_arr = np.zeros(num_portfolios)
                
                # Run simulation
                for i in range(num_portfolios):
                    # Generate random weights
                    if allow_shorting:
                        # Generate weights between min_weight and max_weight
                        weights = np.random.uniform(min_weight, max_weight, num_assets)
                    else:
                        # Generate positive weights only
                        weights = np.random.uniform(0, max_weight, num_assets)
                    
                    # Normalize to sum to 1
                    weights = weights / np.sum(weights)

                    
                    # Save weights
                    all_weights[i, :] = weights
                    
                    # Calculate return and volatility
                    ret_arr[i] = portfolio_return(weights, mean_returns)
                    vol_arr[i] = portfolio_volatility(weights, cov_matrix)
                    
                    # Calculate Sharpe Ratio
                    sharpe_arr[i] = (ret_arr[i] - risk_free_rate) / vol_arr[i]
                
                # Find portfolio with highest Sharpe Ratio
                max_sharpe_idx = sharpe_arr.argmax()
                max_sharpe_allocation = all_weights[max_sharpe_idx, :]
                
                # Find portfolio with minimum volatility
                min_vol_idx = vol_arr.argmin()
                min_vol_allocation = all_weights[min_vol_idx, :]
                
                # Create a dataframe for simulation results
                sim_df = pd.DataFrame({
                    'Return (%)': ret_arr * 100,
                    'Volatility (%)': vol_arr * 100,
                    'Sharpe Ratio': sharpe_arr
                })
                
                # Plot simulation results
                fig3 = px.scatter(
                    sim_df,
                    x='Volatility (%)',
                    y='Return (%)',
                    color='Sharpe Ratio',
                    color_continuous_scale='Viridis',
                    title="Monte Carlo Simulation of Portfolios"
                )
                
                # Add optimal portfolio from optimization
                fig3.add_trace(go.Scatter(
                    x=[optimal_volatility * 100],
                    y=[optimal_return * 100],
                    mode='markers',
                    name='Optimal Portfolio (Optimization)',
                    marker=dict(color='red', size=15, symbol='star')
                ))
                
                # Add max Sharpe portfolio from simulation
                max_sharpe_ret = ret_arr[max_sharpe_idx]
                max_sharpe_vol = vol_arr[max_sharpe_idx]
                
                fig3.add_trace(go.Scatter(
                    x=[max_sharpe_vol * 100],
                    y=[max_sharpe_ret * 100],
                    mode='markers',
                    name='Max Sharpe (Simulation)',
                    marker=dict(color='green', size=15, symbol='diamond')
                ))
                
                # Add min volatility portfolio from simulation
                min_vol_ret = ret_arr[min_vol_idx]
                min_vol_vol = vol_arr[min_vol_idx]
                
                fig3.add_trace(go.Scatter(
                    x=[min_vol_vol * 100],
                    y=[min_vol_ret * 100],
                    mode='markers',
                    name='Min Volatility (Simulation)',
                    marker=dict(color='blue', size=15, symbol='circle')
                ))
                
                fig3.update_layout(height=600)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Display max Sharpe and min volatility portfolios from simulation
                st.subheader("Simulation Results")
                
                # Max Sharpe portfolio
                st.write("Maximum Sharpe Ratio Portfolio (Simulation)")
                max_sharpe_portfolio = pd.DataFrame({
                    'Stock': returns_df.columns,
                    'Weight (%)': [weight * 100 for weight in max_sharpe_allocation]
                })
                
                max_sharpe_portfolio = max_sharpe_portfolio.sort_values('Weight (%)', ascending=False)
                st.write(max_sharpe_portfolio)
                
                st.write(f"Expected Annual Return: {max_sharpe_ret*100:.2f}%")
                st.write(f"Expected Annual Volatility: {max_sharpe_vol*100:.2f}%")
                st.write(f"Sharpe Ratio: {sharpe_arr[max_sharpe_idx]:.2f}")
                
                # Min volatility portfolio
                st.write("Minimum Volatility Portfolio (Simulation)")
                min_vol_portfolio = pd.DataFrame({
                    'Stock': returns_df.columns,
                    'Weight (%)': [weight * 100 for weight in min_vol_allocation]
                })
                
                min_vol_portfolio = min_vol_portfolio.sort_values('Weight (%)', ascending=False)
                st.write(min_vol_portfolio)
                
                st.write(f"Expected Annual Return: {min_vol_ret*100:.2f}%")
                st.write(f"Expected Annual Volatility: {min_vol_vol*100:.2f}%")
                st.write(f"Sharpe Ratio: {sharpe_arr[min_vol_idx]:.2f}")
        else:
            st.error("No data found for the selected stocks or not enough stocks for portfolio analysis. Please check the symbols and try again.")
else:
    st.info("Enter at least two stock symbols and date range, then click 'Fetch Data' to analyze.")

# Footer
st.markdown("---")
st.markdown("Data provided by Yahoo Finance via yfinance")
