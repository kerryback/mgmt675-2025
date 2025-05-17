
import streamlit as st
import pandas as pd
import json
import requests
import os
import re
from datetime import datetime
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="SEC Financial Data Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .error-text {
        color: #F44336;
        font-weight: bold;
    }
    .download-button {
        background-color: #1E88E5;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        text-decoration: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---- CIK Lookup Functions ----

@st.cache_data(ttl=7*24*60*60)  # Cache for 7 days
def download_company_tickers():
    """
    Download the latest company tickers data from the SEC website.
    
    Returns:
        dict: The company tickers data if successful, None otherwise
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error downloading company tickers: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error downloading company tickers: {e}")
        return None

def get_cik_from_ticker(ticker, company_data):
    """
    Get the CIK number for a given ticker symbol.
    
    Args:
        ticker (str): The ticker symbol to look up
        company_data (dict): The company tickers data
        
    Returns:
        str: The CIK number if found, None otherwise
    """
    if not company_data:
        return None
    
    # Normalize ticker to uppercase
    ticker = ticker.upper()
    
    # Search for the ticker in the company data
    for _, company in company_data.items():
        if company['ticker'].upper() == ticker:
            # Found the ticker, return the CIK
            # The CIK is stored as an integer, so convert to string and pad with zeros
            cik_str = str(company['cik_str']).zfill(10)
            return cik_str, company['title']
    
    return None, None

def download_sec_data(cik):
    """
    Download SEC financial data for a given CIK.
    
    Args:
        cik (str): The CIK number
        
    Returns:
        dict: The SEC data if successful, None otherwise
    """
    # Remove leading zeros for the API
    cik_int = int(cik)
    
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_int}.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    try:
        with st.spinner(f"Downloading SEC data for CIK {cik_int}..."):
            response = requests.get(url, headers=headers)
            
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error downloading SEC data: HTTP {response.status_code}")
            if response.status_code == 403:
                st.info("SEC API may be rate limiting requests. Please try again later.")
            return None
    except Exception as e:
        st.error(f"Error downloading SEC data: {e}")
        return None

# ---- SEC JSON Processing Functions ----

def extract_data(data, metric_key, period_type='FY'):
    """
    Extract data for a specific metric and period type (quarterly or annual)
    
    Args:
        data: The JSON data
        metric_key: The US-GAAP metric to extract
        period_type: 'FY' for annual, 'Q' for quarterly
        
    Returns:
        Dictionary mapping periods to values
    """
    if metric_key not in data['facts']['us-gaap']:
        return {}
    
    extracted_data = {}
    
    if 'units' in data['facts']['us-gaap'][metric_key]:
        for unit, values in data['facts']['us-gaap'][metric_key]['units'].items():
            for val in values:
                if 'end' in val and 'val' in val and 'fp' in val:
                    # For quarterly data, we want Q1, Q2, Q3
                    # For annual data, we want FY
                    if (period_type == 'Q' and val['fp'] in ['Q1', 'Q2', 'Q3']) or                        (period_type == 'FY' and val['fp'] == 'FY'):
                        
                        end_date = val['end']
                        year = end_date[:4]
                        
                        # For quarterly data, create a period identifier (e.g., 2010-Q1)
                        if period_type == 'Q':
                            period = f"{year}-{val['fp']}"
                        else:
                            period = year
                        
                        if period not in extracted_data:
                            extracted_data[period] = val['val']
    
    return extracted_data

def process_sec_data(data, progress_bar=None):
    """
    Process SEC data and create annual and quarterly DataFrames
    
    Args:
        data: The SEC JSON data
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        tuple: (annual_df, quarterly_df)
    """
    # Verify the data structure
    if 'facts' not in data or 'us-gaap' not in data['facts']:
        st.error("Error: Invalid SEC JSON data format")
        return None, None
    
    # Get all US-GAAP metrics
    us_gaap_metrics = list(data['facts']['us-gaap'].keys())
    
    # Create annual dataframe with all metrics
    annual_data = {}
    quarterly_data = {}
    
    # Process metrics in batches to avoid memory issues
    batch_size = 50
    total_batches = (len(us_gaap_metrics) + batch_size - 1) // batch_size
    
    for i in range(0, len(us_gaap_metrics), batch_size):
        batch_metrics = us_gaap_metrics[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        if progress_bar:
            progress_bar.progress(batch_num / total_batches, 
                                 text=f"Processing batch {batch_num}/{total_batches}...")
        
        for metric in batch_metrics:
            # Process annual data
            yearly_data = extract_data(data, metric, 'FY')
            for year, value in yearly_data.items():
                if year not in annual_data:
                    annual_data[year] = {}
                annual_data[year][metric] = value
            
            # Process quarterly data
            quarterly_values = extract_data(data, metric, 'Q')
            for quarter, value in quarterly_values.items():
                if quarter not in quarterly_data:
                    quarterly_data[quarter] = {}
                quarterly_data[quarter][metric] = value
    
    # Convert to pandas DataFrames
    annual_df = pd.DataFrame.from_dict(annual_data, orient='index')
    annual_df.index.name = 'Year'
    annual_df.sort_index(inplace=True)
    
    quarterly_df = pd.DataFrame.from_dict(quarterly_data, orient='index')
    quarterly_df.index.name = 'Quarter'
    quarterly_df.sort_index(inplace=True)
    
    return annual_df, quarterly_df

def get_download_link(df, filename, button_text):
    """
    Generate a download link for a DataFrame
    
    Args:
        df: The DataFrame to download
        filename: The filename for the download
        button_text: The text to display on the button
        
    Returns:
        str: HTML for the download link
    """
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{button_text}</a>'
    return href

# ---- Main App ----

def main():
    st.markdown('<h1 class="main-header">SEC Financial Data Extractor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="info-text">Enter a stock ticker symbol to extract financial data from the SEC EDGAR database.</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This app allows you to extract financial data for US public companies "
        "from the SEC's EDGAR database. Enter a ticker symbol to get started."
    )
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        "1. Enter a ticker symbol (e.g., AAPL, MSFT, GOOGL)"

        "2. Click 'Look up CIK'"

        "3. If the CIK is found, click 'Download SEC Data'"

        "4. Once the data is processed, download the annual and quarterly CSV files"
    )
    
    # Input for ticker symbol
    ticker = st.text_input("Enter Ticker Symbol:", placeholder="e.g., AAPL").strip().upper()
    
    if ticker:
        # Look up CIK
        if st.button("Look up CIK"):
            with st.spinner("Looking up CIK..."):
                company_data = download_company_tickers()
                if company_data:
                    cik, company_name = get_cik_from_ticker(ticker, company_data)
                    if cik:
                        st.session_state['cik'] = cik
                        st.session_state['company_name'] = company_name
                        st.success(f"CIK found: {cik}")
                        st.info(f"Company: {company_name}")
                    else:
                        st.error(f"Could not find CIK for ticker: {ticker}")
                else:
                    st.error("Could not download company data from SEC")
    
    # If CIK is in session state, show download button
    if 'cik' in st.session_state:
        st.markdown('<h2 class="sub-header">Download SEC Data</h2>', unsafe_allow_html=True)
        st.info(f"CIK: {st.session_state['cik']} | Company: {st.session_state['company_name']}")
        
        if st.button("Download SEC Data"):
            # Download SEC data
            sec_data = download_sec_data(st.session_state['cik'])
            
            if sec_data:
                st.success("SEC data downloaded successfully")
                
                # Process the data
                st.markdown('<h2 class="sub-header">Processing Data</h2>', unsafe_allow_html=True)
                progress_bar = st.progress(0, text="Starting processing...")
                
                annual_df, quarterly_df = process_sec_data(sec_data, progress_bar)
                
                if annual_df is not None and quarterly_df is not None:
                    # Store the dataframes in session state
                    st.session_state['annual_df'] = annual_df
                    st.session_state['quarterly_df'] = quarterly_df
                    
                    # Display summary
                    st.markdown('<h2 class="sub-header">Data Summary</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Annual Data")
                        st.write(f"Years covered: {annual_df.index.min()} to {annual_df.index.max()}")
                        st.write(f"Number of metrics: {annual_df.shape[1]}")
                        st.write(f"Total data points: {annual_df.size}")
                    
                    with col2:
                        st.markdown("### Quarterly Data")
                        st.write(f"Quarters covered: {quarterly_df.index.min()} to {quarterly_df.index.max()}")
                        st.write(f"Number of metrics: {quarterly_df.shape[1]}")
                        st.write(f"Total data points: {quarterly_df.size}")
                    
                    # Show sample of the data
                    st.markdown("### Sample of Annual Data")
                    st.dataframe(annual_df.head())
                    
                    st.markdown("### Sample of Quarterly Data")
                    st.dataframe(quarterly_df.head())
                    
                    # Download links
                    st.markdown('<h2 class="sub-header">Download CSV Files</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        annual_filename = f"CIK{int(st.session_state['cik'])}_annual.csv"
                        st.markdown(get_download_link(annual_df, annual_filename, "Download Annual Data"), unsafe_allow_html=True)
                    
                    with col2:
                        quarterly_filename = f"CIK{int(st.session_state['cik'])}_quarterly.csv"
                        st.markdown(get_download_link(quarterly_df, quarterly_filename, "Download Quarterly Data"), unsafe_allow_html=True)
                else:
                    st.error("Error processing SEC data")
            else:
                st.error("Error downloading SEC data")

if __name__ == "__main__":
    main()
