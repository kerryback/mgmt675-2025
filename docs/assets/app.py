import streamlit as st
import io
from cost_of_equity_analysis import create_cost_of_equity_deck
import matplotlib.pyplot as plt

def main():
    st.title("Cost of Equity Analysis App")
    
    # Add description
    st.write("""
    This app calculates the cost of equity for any publicly traded company and generates 
    a PowerPoint presentation with the analysis. The analysis includes:
    - Beta calculation with regression plot
    - Risk-free rate from current 3-month T-bill
    - Market risk premium from Fama-French data
    - Final cost of equity calculation
    """)
    
    # Get ticker input
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT, WMT):", "").upper()
    
    if st.button("Generate Analysis"):
        if ticker:
            try:
                st.info(f"Generating analysis for {ticker}...")
                
                # Generate the presentation
                filename, results = create_cost_of_equity_deck(ticker)
                
                # Display results
                st.subheader("Results:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Risk-free Rate", f"{results['risk_free_rate']:.2%}")
                    st.metric("Beta", f"{results['beta']:.2f}")
                
                with col2:
                    st.metric("Market Risk Premium", f"{results['market_risk_premium']:.2%}")
                    st.metric("Cost of Equity", f"{results['cost_of_equity']:.2%}")
                
                # Display the beta plot
                st.subheader("Beta Regression Plot")
                st.pyplot(results['plot'])
                
                # Provide download link for the presentation
                with open(filename, "rb") as file:
                    btn = st.download_button(
                        label="Download PowerPoint Presentation",
                        data=file,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a ticker symbol")

if __name__ == "__main__":
    main()
