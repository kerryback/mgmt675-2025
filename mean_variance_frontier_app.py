
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Mean-Variance Frontier Calculator",
    page_icon="📈",
    layout="wide"
)

# Function to compute and plot the mean-variance frontier
def mean_variance_frontier(expected_returns, std_devs, correlations):
    """
    Compute and plot the mean-variance frontier for risky assets.
    
    Parameters:
    expected_returns (array-like): Expected returns for each asset
    std_devs (array-like): Standard deviations for each asset
    correlations (array-like): Correlation matrix for the assets
    
    Returns:
    tuple: (weights_df, frontier_returns, frontier_stds, fig)
    """
    # Convert inputs to numpy arrays
    expected_returns = np.array(expected_returns)
    std_devs = np.array(std_devs)
    correlations = np.array(correlations)
    
    # Number of assets
    n = len(expected_returns)
    
    # Check dimensions
    if len(std_devs) != n:
        raise ValueError("Length of expected_returns and std_devs must be the same")
    if correlations.shape != (n, n):
        raise ValueError(f"Correlation matrix must be {n}x{n}")
    
    # Compute covariance matrix
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = correlations[i, j] * std_devs[i] * std_devs[j]
    
    # Function to minimize portfolio variance for a given target return
    def portfolio_variance(weights, cov_matrix):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Function to calculate portfolio return
    def portfolio_return(weights, expected_returns):
        return np.sum(weights * expected_returns)
    
    # Constraint: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: no short selling (weights between 0 and 1)
    bounds = tuple((0, 1) for _ in range(n))
    
    # Initial guess
    initial_weights = np.ones(n) / n
    
    # Find minimum variance portfolio
    min_var_result = minimize(
        portfolio_variance, 
        initial_weights, 
        args=(cov_matrix,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    min_var_weights = min_var_result['x']
    min_var_return = portfolio_return(min_var_weights, expected_returns)
    min_var_std = np.sqrt(portfolio_variance(min_var_weights, cov_matrix))
    
    # Find maximum return portfolio (100% in the asset with highest return)
    max_return_idx = np.argmax(expected_returns)
    max_return = expected_returns[max_return_idx]
    max_return_weights = np.zeros(n)
    max_return_weights[max_return_idx] = 1
    max_return_std = std_devs[max_return_idx]
    
    # Generate frontier points
    target_returns = np.linspace(min_var_return, max_return, 100)
    frontier_stds = np.zeros(len(target_returns))
    weights_list = []
    
    for i, target_return in enumerate(target_returns):
        # Add return constraint
        return_constraint = {'type': 'eq', 'fun': lambda x: portfolio_return(x, expected_returns) - target_return}
        constraints_with_return = (constraints, return_constraint)
        
        # Optimize
        result = minimize(
            portfolio_variance, 
            initial_weights, 
            args=(cov_matrix,), 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints_with_return
        )
        
        # If optimization failed, try again with different initial weights
        if not result['success']:
            result = minimize(
                portfolio_variance, 
                np.random.dirichlet(np.ones(n)), 
                args=(cov_matrix,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints_with_return
            )
        
        weights = result['x']
        frontier_stds[i] = np.sqrt(portfolio_variance(weights, cov_matrix))
        weights_list.append(weights)
    
    # Create DataFrame for weights
    weights_df = pd.DataFrame(weights_list, columns=[f'Asset {i+1}' for i in range(n)])
    weights_df['Return'] = target_returns
    weights_df['Risk (Std Dev)'] = frontier_stds
    
    # Plot the frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frontier_stds, target_returns, 'b-', linewidth=3, label='Efficient Frontier')
    
    # Plot individual assets
    ax.scatter(std_devs, expected_returns, c='red', marker='o', s=100, label='Individual Assets')
    
    # Plot minimum variance portfolio
    ax.scatter(min_var_std, min_var_return, c='green', marker='*', s=200, label='Minimum Variance Portfolio')
    
    # Annotate assets
    for i in range(n):
        ax.annotate(f'Asset {i+1}', (std_devs[i], expected_returns[i]), 
                     xytext=(10, 10), textcoords='offset points')
    
    ax.set_title('Mean-Variance Frontier (Risky Assets Only)', fontsize=16)
    ax.set_xlabel('Risk (Standard Deviation)', fontsize=14)
    ax.set_ylabel('Expected Return', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    return weights_df, target_returns, frontier_stds, fig

# Title and description
st.title("📈 Mean-Variance Frontier Calculator")
st.markdown('''
This app calculates and visualizes the mean-variance frontier for a portfolio of risky assets.
Enter the expected returns, standard deviations, and correlation matrix for your assets below.
''')

# Sidebar for inputs
st.sidebar.header("Portfolio Parameters")

# Number of assets
num_assets = st.sidebar.number_input("Number of Assets", min_value=2, max_value=10, value=3, step=1)

# Create tabs for different input sections
tab1, tab2, tab3 = st.tabs(["Expected Returns & Std Devs", "Correlation Matrix", "Results"])

with tab1:
    st.header("Expected Returns and Standard Deviations")
    
    # Create columns for expected returns and standard deviations
    col1, col2 = st.columns(2)
    
    expected_returns = []
    std_devs = []
    
    with col1:
        st.subheader("Expected Returns")
        for i in range(num_assets):
            expected_returns.append(
                st.number_input(
                    f"Expected Return for Asset {i+1}", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.1 + (i * 0.05),
                    step=0.01,
                    format="%.2f",
                    key=f"return_{i}"
                )
            )
    
    with col2:
        st.subheader("Standard Deviations")
        for i in range(num_assets):
            std_devs.append(
                st.number_input(
                    f"Standard Deviation for Asset {i+1}", 
                    min_value=0.01, 
                    max_value=1.0, 
                    value=0.2 + (i * 0.05),
                    step=0.01,
                    format="%.2f",
                    key=f"std_{i}"
                )
            )

with tab2:
    st.header("Correlation Matrix")
    st.markdown("Enter the correlation between each pair of assets (values between -1 and 1).")
    
    # Initialize correlation matrix with ones on the diagonal
    correlations = np.eye(num_assets)
    
    # Create input fields for the lower triangular part of the correlation matrix
    for i in range(num_assets):
        for j in range(i):
            correlations[i, j] = correlations[j, i] = st.number_input(
                f"Correlation between Asset {j+1} and Asset {i+1}",
                min_value=-1.0,
                max_value=1.0,
                value=0.3,  # Default correlation
                step=0.1,
                format="%.2f",
                key=f"corr_{i}_{j}"
            )
    
    # Display the correlation matrix
    st.subheader("Current Correlation Matrix")
    st.dataframe(pd.DataFrame(
        correlations, 
        columns=[f"Asset {i+1}" for i in range(num_assets)],
        index=[f"Asset {i+1}" for i in range(num_assets)]
    ))

with tab3:
    st.header("Mean-Variance Frontier")
    
    if st.button("Calculate Frontier", type="primary"):
        try:
            # Calculate the frontier
            weights_df, frontier_returns, frontier_stds, fig = mean_variance_frontier(
                expected_returns, std_devs, correlations
            )
            
            # Display the plot
            st.pyplot(fig)
            
            # Display portfolio weights
            st.subheader("Portfolio Weights Along the Frontier")
            st.dataframe(weights_df)
            
            # Add download buttons
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=300)
            buffer.seek(0)
            
            st.download_button(
                label="Download Plot",
                data=buffer,
                file_name="mean_variance_frontier.png",
                mime="image/png"
            )
            
            csv_buffer = io.BytesIO()
            weights_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="Download Portfolio Weights (CSV)",
                data=csv_buffer,
                file_name="portfolio_weights.csv",
                mime="text/csv"
            )
            
            # Display key statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Minimum Variance Portfolio")
                min_var_idx = np.argmin(frontier_stds)
                min_var_weights = weights_df.iloc[min_var_idx, :num_assets]
                
                min_var_data = {
                    "Asset": [f"Asset {i+1}" for i in range(num_assets)],
                    "Weight": min_var_weights.values,
                    "Expected Return": [expected_returns[i] for i in range(num_assets)],
                    "Std Dev": [std_devs[i] for i in range(num_assets)]
                }
                
                st.dataframe(pd.DataFrame(min_var_data))
                st.metric("Portfolio Return", f"{frontier_returns[min_var_idx]:.2%}")
                st.metric("Portfolio Risk", f"{frontier_stds[min_var_idx]:.2%}")
            
            with col2:
                st.subheader("Maximum Sharpe Ratio Portfolio")
                # Assuming risk-free rate of 0
                sharpe_ratios = frontier_returns / frontier_stds
                max_sharpe_idx = np.argmax(sharpe_ratios)
                max_sharpe_weights = weights_df.iloc[max_sharpe_idx, :num_assets]
                
                max_sharpe_data = {
                    "Asset": [f"Asset {i+1}" for i in range(num_assets)],
                    "Weight": max_sharpe_weights.values,
                    "Expected Return": [expected_returns[i] for i in range(num_assets)],
                    "Std Dev": [std_devs[i] for i in range(num_assets)]
                }
                
                st.dataframe(pd.DataFrame(max_sharpe_data))
                st.metric("Portfolio Return", f"{frontier_returns[max_sharpe_idx]:.2%}")
                st.metric("Portfolio Risk", f"{frontier_stds[max_sharpe_idx]:.2%}")
                st.metric("Sharpe Ratio", f"{sharpe_ratios[max_sharpe_idx]:.2f}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown('''
This app implements the Markowitz Mean-Variance Portfolio Theory to find the efficient frontier for a set of risky assets.

**Key Concepts:**
- **Mean-Variance Frontier**: The set of portfolios that offer the highest expected return for a given level of risk.
- **Minimum Variance Portfolio**: The portfolio with the lowest possible risk.
- **Maximum Sharpe Ratio Portfolio**: The portfolio with the highest return-to-risk ratio (assuming a risk-free rate of 0).
''')
