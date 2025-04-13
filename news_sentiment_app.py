import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set page title and configuration
st.set_page_config(page_title="News Sentiment Analyzer", layout="wide")
# Get OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.error("OpenAI API key not found in environment variables. Please check your .env file contains OPENAI_API_KEY=your-key-here")
    st.stop()

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

# App title and description
st.title("📰 News Sentiment Analyzer")
st.markdown("This app scrapes headlines from The Guardian and analyzes whether it's a positive or negative news day using ChatGPT-4o.")

# Your OpenAI API key (embedded in the app)
# In a production environment, you should use a more secure method like environment variables or secrets management


# Function to scrape headlines from The Guardian
def scrape_guardian_headlines():
    url = "https://www.theguardian.com/us"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        with st.spinner("Scraping headlines from The Guardian..."):
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find headline elements - The Guardian's structure may change, so we'll try different selectors
            headlines = []
            
            # Look for headline elements with different selectors
            headline_elements = soup.select("a.dcr-lv2z3y")
            if not headline_elements:
                headline_elements = soup.select(".fc-item__title")
            if not headline_elements:
                headline_elements = soup.select(".js-headline-text")
            if not headline_elements:
                headline_elements = soup.select("h3")
                
            # Extract text from headline elements
            for element in headline_elements:
                headline_text = element.get_text().strip()
                if headline_text and len(headline_text) > 10:  # Filter out very short texts
                    headlines.append(headline_text)
                    
            # Remove duplicates while preserving order
            unique_headlines = []
            for headline in headlines:
                if headline not in unique_headlines:
                    unique_headlines.append(headline)
                    
            return unique_headlines[:15]  # Return top 15 headlines
            
    except Exception as e:
        st.error(f"Error scraping The Guardian: {e}")
        return []

# Function to analyze headlines with ChatGPT-4o
def analyze_headlines_with_gpt4o(headlines):
    if not openai_api_key:
        st.error("OpenAI API key not found. Please check your environment variables.")
        return None
    
    # Create prompt for analysis
    prompt = f"""
    Here are today's ({datetime.now().strftime('%Y-%m-%d')}) headlines from The Guardian:
    
    {chr(10).join(['- ' + headline for headline in headlines])}
    
    Based on these headlines, is this a negative news day or a positive news day? Please explain your reasoning.
    """
    
    try:
        with st.spinner("Analyzing headlines with ChatGPT-4o..."):
            # Call the OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes news headlines."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content
            
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None

# Main app functionality
if st.button("Analyze Today's News", type="primary", use_container_width=True):
    # Scrape headlines
    headlines = scrape_guardian_headlines()
    
    if headlines:
        # Display headlines
        st.subheader("📋 Today's Headlines")
        for i, headline in enumerate(headlines, 1):
            st.write(f"{i}. {headline}")
        
        # Analyze headlines
        analysis = analyze_headlines_with_gpt4o(headlines)
        
        if analysis:
            # Display analysis
            st.subheader("🤖 ChatGPT-4o Analysis")
            st.markdown(analysis)
            
            # Add a visual indicator of sentiment
            if "negative" in analysis.lower() and "positive" not in analysis.lower():
                st.error("📉 NEGATIVE NEWS DAY")
            elif "positive" in analysis.lower() and "negative" not in analysis.lower():
                st.success("📈 POSITIVE NEWS DAY")
            else:
                st.warning("📊 MIXED NEWS DAY")
    else:
        st.warning("No headlines were scraped. Please try again later.")

# Add footer
st.markdown("---")
st.markdown("Created with Streamlit and OpenAI's GPT-4o")
