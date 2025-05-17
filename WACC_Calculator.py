import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure required packages are installed
install_if_missing('nltk')
install_if_missing('textblob')
install_if_missing('gnews')
install_if_missing('yfinance')
install_if_missing('pandas')
install_if_missing('matplotlib')
install_if_missing('streamlit')
install_if_missing('requests')

# For nltk and textblob, make sure the required corpora are downloaded
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


import pandas as pd
import yfinance as yf
import nltk
# Optional: Check and download 'punkt' only if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
def get_sp500_tickers():
    tables = pd.read_html(sp500_url)
    df = tables[0]
    return df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']]

def get_peers_by_industry(ticker):
    info = yf.Ticker(ticker).info
    industry = info.get('industry', None)
    sector = info.get('sector', None)
    if not industry:
        return [], None, None
    sp500_df = get_sp500_tickers()
    main_row = sp500_df[sp500_df['Symbol'] == ticker]
    if not main_row.empty:
        sub_industry = main_row.iloc[0]['GICS Sub-Industry']
        peer_rows = sp500_df[(sp500_df['GICS Sub-Industry'] == sub_industry) & (sp500_df['Symbol'] != ticker)]
        peer_tickers = peer_rows['Symbol'].tolist()[:5]
        return peer_tickers, industry, sector
    else:
        peer_tickers = []
        for sym in sp500_df['Symbol']:
            try:
                peer_info = yf.Ticker(sym).info
                if peer_info.get('industry', None) == industry and sym != ticker:
                    peer_tickers.append(sym)
            except Exception:
                continue
            if len(peer_tickers) >= 5:
                break
        return peer_tickers, industry, sector

import yfinance as yf
import pandas as pd


def get_peer_metrics(tickers, main_ticker, risk_free_rate, equity_market_premium, tax_rate):
    data = []
    for t in [main_ticker] + tickers:
        try:
            info = yf.Ticker(t).info
            company_name = info.get('shortName', t)
            beta = info.get('beta', None)
            market_cap = info.get('marketCap', None)
            total_debt = info.get('totalDebt', None)
            if not all([beta, market_cap, total_debt]):
                continue
            cost_of_equity = risk_free_rate + beta * equity_market_premium
            V = market_cap + total_debt
            weight_equity = market_cap / V
            weight_debt = total_debt / V
            interest_rate_on_debt = info.get('yield', 0.04)
            wacc = weight_equity * cost_of_equity + weight_debt * interest_rate_on_debt * (1 - tax_rate)
            data.append({
                'Ticker': t,
                'Company Name': company_name,
                'Beta': beta,
                'Market Cap ($B)': market_cap / 1e9,
                'Total Debt ($B)': total_debt / 1e9,
                'WACC (%)': wacc * 100
            })
        except Exception:
            continue
    return pd.DataFrame(data)
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import pandas as pd
import requests
st.set_page_config(page_title="WACC Calculator", layout="wide")


def fetch_risk_free_rate(years):
    """Fetch Treasury yield for specified years"""
    try:
        # Map years to corresponding Treasury tickers
        treasury_mapping = {
            1: '^IRX',    # 13-week Treasury Bill
            2: '^FVX',    # 5-year Treasury Note (closest to 2-year)
            5: '^FVX',    # 5-year Treasury Note
            10: '^TNX',   # 10-year Treasury Note
            20: '^TYX',   # 30-year Treasury Bond (closest to 20-year)
            30: '^TYX'    # 30-year Treasury Bond
        }

        # Find the closest year in our mapping
        available_years = list(treasury_mapping.keys())
        closest_year = min(available_years, key=lambda x: abs(x - years))
        ticker = treasury_mapping[closest_year]

        risk_ticker = yf.Ticker(ticker)
        risk_rate = risk_ticker.history(period='1d')['Close'].iloc[-1]
        return risk_rate / 100  # Convert to decimal
    except Exception as e:
        print(f"Error fetching treasury rate: {e}")
        return 0.025  # Default to 2.5% if fetch fails

def fetch_market_premium():
    """Fetch equity risk premium - using average historical data"""
    return 0.055  # Historical average of 5.5%

def fetch_corporate_bond_rate(ticker):
    """Attempt to fetch corporate bond yield from company info"""
    try:
        ticker_info = yf.Ticker(ticker).info
        # Try to get cost of debt from financial data
        if 'lastFiscalYearInterestExpense' in ticker_info and 'totalDebt' in ticker_info:
            if ticker_info['totalDebt'] > 0:
                return abs(ticker_info['lastFiscalYearInterestExpense'] / ticker_info['totalDebt'])
    except Exception:
        pass
    return fetch_risk_free_rate(10)  # Fallback to 10-year Treasury rate if fetch fails

def fetch_tax_rate(ticker):
    """Fetch effective tax rate from financial statements"""
    try:
        ticker_info = yf.Ticker(ticker).info
        if 'effectiveTaxRate' in ticker_info:
            tax_rate = ticker_info['effectiveTaxRate']
            if isinstance(tax_rate, (int, float)) and 0 <= tax_rate <= 1:
                return tax_rate
    except Exception:
        pass
    return 0.21  # Default to 21% (US corporate tax rate) if fetch fails

def calculate_wacc(ticker_symbol, risk_free_rate, equity_market_premium, interest_rate_on_debt, tax_rate, manual_market_cap=None, manual_beta=None):
    try:
        # Get ticker data
        ticker = yf.Ticker(ticker_symbol)

        # Get market cap
        if manual_market_cap:
            market_cap = float(manual_market_cap) * 1e9  # Convert billions to actual value
        else:
            market_cap = ticker.info.get('marketCap', 0)
            if market_cap == 0:
                return None, None, None, None, None, None, "Could not fetch market cap. Please enter manually."

        # Get beta
        if manual_beta:
            beta = float(manual_beta)
        else:
            beta = ticker.info.get('beta', None)
            if beta is None:
                return None, None, None, None, None, None, "Could not fetch beta. Please enter manually."

        # Get total debt
        try:
            total_debt = ticker.info.get('totalDebt', 0)
            if total_debt == 0:
                # Try alternative fields
                total_debt = (
                    ticker.info.get('shortLongTermDebt', 0) +
                    ticker.info.get('longTermDebt', 0)
                )
        except:
            total_debt = 0

        # Calculate firm value
        V = market_cap + total_debt

        if V == 0:
            return None, None, None, None, None, None, "Error: Total firm value is zero"

        # Calculate cost of equity using CAPM
        cost_of_equity = risk_free_rate + beta * equity_market_premium

        # Calculate WACC
        wacc = (market_cap/V) * cost_of_equity + (total_debt/V) * interest_rate_on_debt * (1 - tax_rate)

        return market_cap, beta, total_debt, V, cost_of_equity, wacc, None

    except Exception as e:
        return None, None, None, None, None, None, f"Error: {str(e)}"
    
# --- Sentiment Analysis and News Section ---

import requests


# --- Simplified Sentiment Section ---
from textblob import TextBlob

from textblob import TextBlob
from gnews import GNews

def get_recent_news(ticker, n=5):
    google_news = GNews(language='en', country='US', max_results=n)
    query = f"{ticker} stock"
    results = google_news.get_news(query)
    news_items = []
    for article in results:
        title = article.get('title', '')
        link = article.get('url', '')
        description = article.get('description', '')
        if title and link:
            full_text = f"{title}. {description}"
            news_items.append((title, link, full_text))
    return news_items

def analyze_news_sentiment(ticker):
    news = get_recent_news(ticker)
    if not news:
        print('No news found for', ticker)
        return 'neutral', []

    sentiments = []
    total_score = 0
    for title, link, text in news:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        total_score += polarity
        if polarity > 0.1:
            label = 'Positive'
        elif polarity < -0.1:
            label = 'Negative'
        else:
            label = 'Neutral'
        sentiments.append((title, link, label, polarity))

    avg_score = total_score / len(sentiments) if sentiments else 0
    if avg_score > 0.1:
        overall = 'positive'
    elif avg_score < -0.1:
        overall = 'negative'
    else:
        overall = 'neutral'

    return overall, sentiments

def news_sentiment_recommendation(ticker):
    overall, details = analyze_news_sentiment(ticker)
    print('Recent news sentiment for', ticker, ':', overall)
    for title, sentiment in details:
        print(' -', title, '=>', sentiment)
    if overall == 'positive':
        print('Based on recent news, the stock price is likely to rise.')
    elif overall == 'negative':
        print('Based on recent news, the stock price is likely to fall.')
    else:
        print('Based on recent news, the stock price is likely to remain stable.')

def main():
    st.markdown("<h3 style='font-size: 18px;'>Created by Seif Moursy, Bret Sharman, Mauricio Parilli, Rudo Ndamba & Louis Delaura for MGMT 675, JGSB, Rice University, 2025.</h3>", unsafe_allow_html=True)
    st.markdown("---")

    st.title("WACC Calculator")
    st.write("Calculate the Weighted Average Cost of Capital (WACC) for any publicly traded company")

    # Input section
    st.sidebar.header("Input Parameters")

    ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()

    # Add toggle for auto-fetch vs manual input
    auto_fetch = st.sidebar.checkbox("Auto-fetch rates", value=True)

    if auto_fetch:
        # Fetch all rates automatically
        years = st.sidebar.selectbox(
            'Treasury Bond Years',
            options=[1, 2, 5, 10, 20, 30],
            index=3  # Default to 10-year
        )
        risk_free_rate = fetch_risk_free_rate(years)
        equity_market_premium = fetch_market_premium()
        interest_rate_on_debt = fetch_corporate_bond_rate(ticker_symbol)
        tax_rate = fetch_tax_rate(ticker_symbol)

        # Display the fetched rates (read-only)
        st.sidebar.subheader("Fetched Rates (Auto)")
        st.sidebar.text(f"{years}-Year Treasury Rate: {risk_free_rate:.2%}")
        st.sidebar.text(f"Equity Market Premium: {equity_market_premium:.2%}")
        st.sidebar.text(f"Interest Rate on Debt: {interest_rate_on_debt:.2%}")
        st.sidebar.text(f"Tax Rate: {tax_rate:.2%}")
    else:
        # Manual input sliders
        st.sidebar.subheader("Manual Rate Inputs")
        risk_free_rate = st.sidebar.slider(
            "Risk-free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=fetch_risk_free_rate(10)*100
        ) / 100

        equity_market_premium = st.sidebar.slider(
            "Equity Market Premium (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.5
        ) / 100

        interest_rate_on_debt = st.sidebar.slider(
            "Interest Rate on Debt (%)",
            min_value=0.0,
            max_value=15.0,
            value=fetch_risk_free_rate(10)*100
        ) / 100

        tax_rate = st.sidebar.slider(
            "Tax Rate (%)",
            min_value=0.0,
            max_value=40.0,
            value=21.0
        ) / 100

    # Optional manual inputs for market cap and beta
    st.sidebar.subheader("Optional Manual Overrides")
    manual_market_cap = st.sidebar.text_input("Market Cap (billions USD)", value="")
    manual_beta = st.sidebar.text_input("Beta", value="")

    if st.sidebar.button("Calculate WACC"):
        market_cap, beta, total_debt, V, cost_of_equity, wacc, error = calculate_wacc(
            ticker_symbol,
            risk_free_rate,
            equity_market_premium,
            interest_rate_on_debt,
            tax_rate,
            manual_market_cap,
            manual_beta
        )

        if error:
            st.error(error)
        else:
            st.markdown(f"<p style='text-align: center; font-size: 60px; font-weight: bold; color: #1f77b4;'>WACC: {wacc:.2%}</p>", unsafe_allow_html=True)

            st.header(f"WACC Analysis for {ticker_symbol}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Market Data")
                st.write(f"Market Cap: ${market_cap/1e9:.2f}B")
                st.write(f"Beta: {beta:.2f}")
                st.write(f"Total Debt: ${total_debt/1e9:.2f}B")
                st.write(f"Firm Value: ${V/1e9:.2f}B")

            with col2:
                st.subheader("Calculated Components")
                st.write(f"Cost of Equity: {cost_of_equity:.2%}")
                st.write(f"Weight of Equity: {market_cap/V:.2%}")
                st.write(f"Weight of Debt: {total_debt/V:.2%}")

            
            # Create and display capital structure charts
            fig = create_capital_structure_charts(market_cap, total_debt, cost_of_equity, wacc, tax_rate, interest_rate_on_debt)
            st.pyplot(fig)

            # --- Peer Analysis Section ---
            peer_tickers, industry, sector = get_peers_by_industry(ticker_symbol)
            if peer_tickers:
                st.subheader(f"Peer Analysis: {industry}")
                peer_df = get_peer_metrics(peer_tickers, ticker_symbol, risk_free_rate, equity_market_premium, tax_rate)
                st.dataframe(peer_df)
                # Professional color palette
                color_wacc = '#2E5C8A'      # Deep navy blue
                color_beta = '#C41E3A'      # Corporate red
                color_mcap = '#485C70'      # Steel gray
                color_debt = '#A9A9A9'      # Muted gray
                fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='none')
                peer_df.plot.bar(x='Ticker', y='WACC (%)', ax=axes[0,0], color=color_wacc, legend=False, alpha=0.85)
                axes[0,0].set_title('WACC (%)')
                peer_df.plot.bar(x='Ticker', y='Beta', ax=axes[0,1], color=color_beta, legend=False, alpha=0.85)
                axes[0,1].set_title('Beta')
                peer_df.plot.bar(x='Ticker', y='Market Cap ($B)', ax=axes[1,0], color=color_mcap, legend=False, alpha=0.85)
                axes[1,0].set_title('Market Cap ($B)')
                peer_df.plot.bar(x='Ticker', y='Total Debt ($B)', ax=axes[1,1], color=color_debt, legend=False, alpha=0.85)
                axes[1,1].set_title('Total Debt ($B)')
                for ax in axes.flat:
                    ax.set_facecolor('none')
                    ax.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            plt.close()
            

            
            # --- News Sentiment Section ---
            
            st.subheader("Recent News Sentiment Analysis")
            overall, details = analyze_news_sentiment(ticker_symbol)
            st.write(f"**Overall Sentiment:** {overall.capitalize()}")
            st.markdown("**Top News Headlines & Sentiments:**")
            for title, link, sentiment, polarity in details:
                st.markdown(f"- [{title}]({link}) — *{sentiment}* (score: {polarity:.2f})")

            st.subheader("WACC Formula")
            st.latex(r'''WACC = \left(\frac{E}{V}\right)r_E + \left(\frac{D}{V}\right)r_D(1-t)''')
            st.write("Where:")
            st.write("- E = Market Cap (Equity)")
            st.write("- D = Total Debt")
            st.write("- V = Firm Value (E + D)")
            st.write("- r_E = Cost of Equity (Risk-free rate + Beta × Equity Market Premium)")
            st.write("- r_D = Interest Rate on Debt")
            st.write("- t = Tax Rate")





def create_capital_structure_charts(market_cap, total_debt, cost_of_equity, wacc, tax_rate, interest_rate_on_debt):
    # Professional financial industry color scheme
    equity_blue = '#2E5C8A'      # Deep navy blue
    debt_gray = '#485C70'        # Steel gray
    accent_color = '#C41E3A'     # Corporate red for WACC line
    background_color = '#FFFFFF'  # Clean white background
    text_color = '#333333'       # Dark gray for text
    grid_color = '#E5E5E5'       # Light gray for grid
    
    # Set style parameters
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'axes.grid': True,
        'grid.color': grid_color,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
    })
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor(background_color)
    
    # Set background for both plots
    for ax in [ax1, ax2]:
        ax.set_facecolor(background_color)
        
    # Pie Chart - Flat design
    values = [market_cap, total_debt]
    labels = ['Equity', 'Debt']
    colors = [equity_blue, debt_gray]
    
    # Create pie chart with minimal effects
    wedges, texts, autotexts = ax1.pie(values, 
                                      labels=labels, 
                                      colors=colors, 
                                      autopct='%1.1f%%',
                                      startangle=90)
    
    # Enhance pie chart text
    plt.setp(autotexts, size=10, weight='bold', color='white')
    plt.setp(texts, size=11, color=text_color)
    ax1.set_title('Capital Structure', pad=20, size=14, weight='bold', color=text_color)
    
    # Bar Chart
    equity_contribution = (market_cap / (market_cap + total_debt)) * cost_of_equity
    debt_contribution = (total_debt / (market_cap + total_debt)) * (1 - tax_rate) * interest_rate_on_debt
    
    contributions = [equity_contribution, debt_contribution]
    bars = ax2.bar(labels, contributions, color=colors, width=0.6)
    
    # Enhance bar chart
    ax2.set_title('WACC Components', pad=20, size=14, weight='bold', color=text_color)
    ax2.set_ylabel('Contribution to WACC', size=11, color=text_color)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', size=10,
                color=text_color)
    
    # Add total WACC line with refined styling
    ax2.axhline(y=wacc, color=accent_color, linestyle='--', linewidth=1.5, label='Total WACC')
    ax2.text(ax2.get_xlim()[1], wacc, f' Total WACC: {wacc:.2%}', 
             va='bottom', ha='right', color=accent_color, weight='bold')
    
    # Enhance legend with minimal styling
    legend = ax2.legend(frameon=True, facecolor=background_color, 
                       edgecolor=grid_color, fontsize=10)
    
    # Add subtle grid to bar chart only
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(grid_color)
    ax2.spines['bottom'].set_color(grid_color)
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    return fig

if __name__ == "__main__":
    main()
