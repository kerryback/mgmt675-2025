import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Function to calculate maximum feasible withdrawal
def calculate_max_withdrawal(
    current_balance,
    years_before_retirement,
    annual_deposit,
    rate_of_return,
    withdrawal_years,
):
    future_balance = current_balance * (1 + rate_of_return) ** years_before_retirement

    if annual_deposit != 0:
        fv_deposits = (
            annual_deposit
            * ((1 + rate_of_return) ** years_before_retirement - 1)
            / rate_of_return
        )
    else:
        fv_deposits = 0

    if rate_of_return != 0:
        pv_factor = (1 - (1 + rate_of_return) ** -withdrawal_years) / rate_of_return
    else:
        pv_factor = withdrawal_years

    max_withdrawal = (future_balance + fv_deposits) / pv_factor
    return max_withdrawal


# Function to calculate yearly balances for simulation
def calculate_yearly_balances(
    current_balance,
    years_before_retirement,
    annual_deposit,
    withdrawal_years,
    withdrawal_amount,
    returns,
    borrowing_rate,
):
    total_years = years_before_retirement + withdrawal_years
    balance = current_balance
    yearly_results = []

    for year in range(total_years):
        beginning_balance = balance
        rate = returns[year] if beginning_balance > 0 else borrowing_rate
        gain_loss = beginning_balance * rate
        cash_flow = (
            annual_deposit if year < years_before_retirement else -withdrawal_amount
        )
        balance = beginning_balance + gain_loss + cash_flow

        yearly_results.append(
            {
                "Year": year + 1,
                "Beginning Balance": beginning_balance,
                "Return": rate,
                "Gain/Loss": gain_loss,
                "Deposit/Withdrawal": cash_flow,
                "Ending Balance": balance,
            }
        )

    return yearly_results


st.title("Retirement Planning Calculator")

# Part 1: Maximum Withdrawal Calculator
st.header("Part 1: Calculate Maximum Feasible Withdrawal")

col1, col2 = st.columns(2)

with col1:
    current_balance = st.number_input(
        "Current Balance ($)", min_value=0.0, value=50000.0
    )
    years_before_retirement = st.number_input(
        "Years Before Retirement", min_value=0, value=5
    )
    annual_deposit = st.number_input("Annual Deposit ($)", min_value=0.0, value=10000.0)

with col2:
    rate_of_return = st.number_input(
        "Expected Rate of Return",
        min_value=-1.0,
        max_value=1.0,
        value=0.10,
        format="%.3f",
    )
    withdrawal_years = st.number_input("Years to Withdraw", min_value=1, value=5)

if st.button("Calculate Maximum Withdrawal"):
    max_withdrawal = calculate_max_withdrawal(
        current_balance=current_balance,
        years_before_retirement=years_before_retirement,
        annual_deposit=annual_deposit,
        rate_of_return=rate_of_return,
        withdrawal_years=withdrawal_years,
    )
    st.success(f"Maximum Annual Withdrawal: ${max_withdrawal:,.2f}")

# Part 2: Monte Carlo Simulation
st.header("Part 2: Monte Carlo Simulation")

col3, col4 = st.columns(2)

with col3:
    mean_return = st.number_input(
        "Expected Mean Return", min_value=-1.0, max_value=1.0, value=0.10, format="%.3f"
    )
    std_dev = st.number_input(
        "Standard Deviation", min_value=0.0, max_value=1.0, value=0.20, format="%.3f"
    )
    withdrawal_amount = st.number_input(
        "Desired Withdrawal Amount ($)", min_value=0.0, value=35000.0
    )

with col4:
    borrowing_rate = st.number_input(
        "Borrowing Rate", min_value=0.0, max_value=1.0, value=0.10, format="%.3f"
    )
    n_simulations = st.number_input(
        "Number of Simulations", min_value=100, max_value=10000, value=1000
    )

if st.button("Run Monte Carlo Simulation"):
    # Create progress bar
    progress_bar = st.progress(0)

    # Initialize arrays to store results
    total_years = years_before_retirement + withdrawal_years
    all_balances = np.zeros((n_simulations, total_years))

    # Run simulations
    for i in range(n_simulations):
        # Generate random returns
        returns = np.random.normal(mean_return, std_dev, total_years)

        # Calculate yearly balances
        yearly_results = calculate_yearly_balances(
            current_balance=current_balance,
            years_before_retirement=years_before_retirement,
            annual_deposit=annual_deposit,
            withdrawal_years=withdrawal_years,
            withdrawal_amount=withdrawal_amount,
            returns=returns,
            borrowing_rate=borrowing_rate,
        )

        # Store ending balances
        all_balances[i] = [result["Ending Balance"] for result in yearly_results]

        # Update progress bar
        progress_bar.progress((i + 1) / n_simulations)

    # Calculate statistics
    median_balance = np.median(all_balances, axis=0)
    mean_balance = np.mean(all_balances, axis=0)
    final_balances = all_balances[:, -1]

    # Create figures
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    years = range(1, total_years + 1)
    ax1.plot(years, median_balance, label="Median Balance", linewidth=2)
    ax1.plot(years, mean_balance, label="Mean Balance", linewidth=2)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Account Balance ($)")
    ax1.set_title("Account Balance Projections")
    ax1.legend()
    ax1.grid(True)

    # Format y-axis with dollar signs and commas
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=final_balances, bins=50, ax=ax2)
    ax2.set_xlabel("Final Account Balance ($)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Final Account Balances")

    # Format x-axis with dollar signs and commas
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    st.pyplot(fig2)

    # Display summary statistics
    st.write("Summary Statistics for Final Balance:")
    st.write(f"Median: ${np.median(final_balances):,.2f}")
    st.write(f"Mean: ${np.mean(final_balances):,.2f}")
    st.write(f"Standard Deviation: ${np.std(final_balances):,.2f}")
    st.write(f"Probability of Negative Balance: {(final_balances < 0).mean():.1%}")
