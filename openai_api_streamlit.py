import streamlit as st
import openai 

from dotenv import load_dotenv
import os
from openai import OpenAI

# Load variables from .env file into environment
load_dotenv()

# Create OpenAI client
try:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    st.error("OpenAI API key not found! Please check your .env file contains: OPENAI_API_KEY=your-api-key")
    raise

def main():
    st.title("Finance Assistant with OpenAI") 
    
    # User input area
    user_query = st.text_input("Ask your finance question:") 
    
    if user_query:
        # Generate response using OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Answer this finance question: {user_query}"}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        # Display response 
        st.write(response.choices[0].message.content)

if __name__ == "__main__":
    main()