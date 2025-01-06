import os
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict

# Load environment variables
load_dotenv()

# Google API Key retrieval
def get_api_key():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        if 'GOOGLE_API_KEY' in st.secrets:
            api_key = st.secrets['GOOGLE_API_KEY']
    return api_key

class CSVEnrichmentAgent:
    def __init__(self):
        api_key = get_api_key()
        if not api_key:
            st.error("Google API Key not found. Please set it in your .env file or Streamlit secrets.")
            st.stop()
            
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def analyze_columns(self, columns: List[str]) -> str:
        prompt = f"""Analyze these columns and suggest potential insights:
        Columns: {columns}
        Provide analysis in this format:
        1. Data Overview
        2. Potential Insights
        3. Recommended Visualizations"""
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def suggest_enrichments(self, columns: List[str], sample_data: str) -> str:
        prompt = f"""Given these columns and sample data, suggest enrichment opportunities:
        Columns: {columns}
        Sample Data: {sample_data}
        Provide specific enrichment suggestions that could add value to this dataset."""
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def generate_insights(self, df: pd.DataFrame) -> Dict:
        # Generate basic statistics
        numeric_stats = df.describe()
        missing_values = df.isnull().sum()
        
        return {
            "statistics": numeric_stats,
            "missing_values": missing_values,
            "row_count": len(df),
            "column_count": len(df.columns)
        }
    
    def chat_with_csv(self, query: str, df: pd.DataFrame) -> str:
        sample_data = df.head().to_string()
        prompt = f"""Using this sample data, answer the following question:
        Sample Data: {sample_data}
        Question: {query}
        Provide a concise, insightful answer based on the data."""
        
        response = self.model.generate_content(prompt)
        return response.text

# Upload file to Google Cloud Storage
def upload_to_gcs(bucket_name, file):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file.name)
        blob.upload_from_string(file.read(), content_type="text/csv")
        return f"gs://{bucket_name}/{file.name}"
    except Exception as e:
        st.error(f"Failed to upload file to Google Cloud Storage: {e}")
        return None

def main():
    st.set_page_config(page_title="CSV Analysis Tool", layout="wide")
    st.title("CSV Analysis Tool with Google Cloud Integration and AI Enrichment")
    st.write("Upload a CSV file for analysis, storage in Google Cloud, and AI-powered enrichment.")

    # File upload
    uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])
    
    if uploaded_file:
        # Upload file to Google Cloud Storage
        bucket_name = "cropyieldprediction"  # Your GCS bucket name
        with st.spinner("Uploading to Google Cloud Storage..."):
            gcs_path = upload_to_gcs(bucket_name, uploaded_file)
            if gcs_path:
                st.success(f"File successfully uploaded to: {gcs_path}")
        
        # Load data into pandas DataFrame
        data = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # Sidebar for navigation
        with st.sidebar:
            page = st.radio("Select Page", ["Data Overview", "AI Analysis", "Enrichment Suggestions", "Visualizations", "Chat with CSV"])

        # Initialize the agent
        agent = CSVEnrichmentAgent()

        # Data Overview Page
        if page == "Data Overview":
            st.header("Data Overview")
            st.write("First few rows of your data:")
            st.dataframe(data.head())
            
            insights = agent.generate_insights(data)
            st.subheader("Basic Statistics")
            st.write(insights["statistics"])
            
            st.subheader("Missing Values Analysis")
            st.bar_chart(insights["missing_values"])

        # AI Analysis Page
        elif page == "AI Analysis":
            st.header("AI Analysis of Columns")
            with st.spinner("Generating AI analysis..."):
                analysis = agent.analyze_columns(data.columns.tolist())
                st.write(analysis)

        # Enrichment Suggestions Page
        elif page == "Enrichment Suggestions":
            st.header("Enrichment Suggestions")
            with st.spinner("Generating enrichment suggestions..."):
                sample_data = data.head().to_string()
                suggestions = agent.suggest_enrichments(data.columns.tolist(), sample_data)
                st.write(suggestions)

        # Visualizations Page
        elif page == "Visualizations":
            st.header("Visualizations")
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            # Visualization type selector
            viz_type = st.selectbox("Choose visualization type", ["Scatter Plot", "Bar Chart", "Line Chart", "Box Plot"])

            if viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X axis", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Select Y axis", numeric_cols, key="scatter_y")
                fig = px.scatter(data, x=x_col, y=y_col)
                st.plotly_chart(fig)

            elif viz_type == "Bar Chart" and len(categorical_cols) > 0:
                x_col = st.selectbox("Select category", categorical_cols)
                if len(numeric_cols) > 0:
                    y_col = st.selectbox("Select value", numeric_cols)
                    fig = px.bar(data, x=x_col, y=y_col)
                    st.plotly_chart(fig)

            elif viz_type == "Line Chart" and len(numeric_cols) > 0:
                y_col = st.selectbox("Select value", numeric_cols)
                fig = px.line(data, y=y_col)
                st.plotly_chart(fig)

            elif viz_type == "Box Plot" and len(numeric_cols) > 0:
                y_col = st.selectbox("Select value", numeric_cols)
                if len(categorical_cols) > 0:
                    x_col = st.selectbox("Select category (optional)", categorical_cols)
                    fig = px.box(data, x=x_col, y=y_col)
                else:
                    fig = px.box(data, y=y_col)
                st.plotly_chart(fig)

        # Chat with CSV Page
        elif page == "Chat with CSV":
            st.header("Chat with CSV")
            query = st.text_input("Ask a question about your data:")
            if query:
                with st.spinner("Generating response..."):
                    response = agent.chat_with_csv(query, data)
                    st.write(response)

if __name__ == "__main__":
    main()
