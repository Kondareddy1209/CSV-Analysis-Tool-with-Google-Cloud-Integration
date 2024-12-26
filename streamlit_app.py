import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\Users\Konda Reddy\Documents\GitHub\gen-lang-client-0298324082-8eef79011259.json"  # Update with your credentials path

# Function to upload file to Google Cloud Storage
def upload_to_gcs(bucket_name, file):
    """
    Uploads a file to Google Cloud Storage.
    Args:
        bucket_name (str): Name of the GCS bucket.
        file: The uploaded file from Streamlit's file_uploader.
    Returns:
        str: The GCS file path.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file.name)
        blob.upload_from_string(file.read(), content_type="text/csv")
        return f"gs://{bucket_name}/{file.name}"
    except Exception as e:
        st.error(f"Failed to upload file to Google Cloud Storage: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="CSV Analysis Tool", layout="wide")
st.title("CSV Analysis Tool with Google Cloud Integration")
st.write("Upload a CSV file for analysis and storage in Google Cloud Storage.")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV File", type=["csv"])

if uploaded_file:
    # Upload file to GCS
    bucket_name = "cropyieldprediction"  # Replace with your bucket name
    with st.spinner("Uploading to Google Cloud Storage..."):
        gcs_path = upload_to_gcs(bucket_name, uploaded_file)
        if gcs_path:
            st.success(f"File successfully uploaded to: {gcs_path}")

    # Load the uploaded file into a Pandas DataFrame
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Filtering data
    st.subheader("Filter Data")
    filter_col = st.selectbox("Select a column to filter", data.columns)
    unique_values = data[filter_col].unique()
    selected_values = st.multiselect(f"Select values to filter '{filter_col}':", unique_values)

    if selected_values:
        filtered_data = data[data[filter_col].isin(selected_values)]
        st.dataframe(filtered_data)
    else:
        filtered_data = data

    # Data Visualization: Bar Chart
    st.subheader("Bar Chart")
    x_axis = st.selectbox("Select X-axis column for Bar Chart", data.columns)
    y_axis = st.selectbox("Select Y-axis column for Bar Chart", data.columns)

    if pd.api.types.is_numeric_dtype(filtered_data[y_axis]):
        st.write(f"Visualizing {y_axis} vs {x_axis}")
        chart_data = filtered_data[[x_axis, y_axis]].groupby(x_axis).mean()
        st.bar_chart(chart_data)
    else:
        st.warning(f"Cannot visualize non-numeric data in the {y_axis} column.")

    # Data Visualization: Pie Chart
    st.subheader("Pie Chart")
    pie_column = st.selectbox("Select a column for Pie Chart", data.columns)

    if filtered_data[pie_column].dtype == 'object':
        pie_data = filtered_data[pie_column].value_counts()
        fig, ax = plt.subplots()
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)
    else:
        st.warning("Please select a categorical column for Pie Chart.")

    # Data Visualization: Line Graph
    st.subheader("Line Graph")
    line_x_axis = st.selectbox("Select X-axis column for Line Graph", data.columns)
    line_y_axis = st.selectbox("Select Y-axis column for Line Graph", data.columns)

    if pd.api.types.is_numeric_dtype(filtered_data[line_x_axis]) and pd.api.types.is_numeric_dtype(filtered_data[line_y_axis]):
        st.write(f"Visualizing {line_y_axis} vs {line_x_axis}")
        fig, ax = plt.subplots()
        ax.plot(filtered_data[line_x_axis], filtered_data[line_y_axis], marker='o')
        ax.set_xlabel(line_x_axis)
        ax.set_ylabel(line_y_axis)
        ax.set_title(f"{line_y_axis} vs {line_x_axis}")
        st.pyplot(fig)
    else:
        st.warning("Both X-axis and Y-axis need to be numeric for a line graph.")

    # Data Visualization: Scatter Plot
    st.subheader("Scatter Plot")
    scatter_x_axis = st.selectbox("Select X-axis column for Scatter Plot", data.columns)
    scatter_y_axis = st.selectbox("Select Y-axis column for Scatter Plot", data.columns)

    if pd.api.types.is_numeric_dtype(filtered_data[scatter_x_axis]) and pd.api.types.is_numeric_dtype(filtered_data[scatter_y_axis]):
        st.write(f"Scatter plot for {scatter_y_axis} vs {scatter_x_axis}")
        fig, ax = plt.subplots()
        ax.scatter(filtered_data[scatter_x_axis], filtered_data[scatter_y_axis], alpha=0.7)
        ax.set_xlabel(scatter_x_axis)
        ax.set_ylabel(scatter_y_axis)
        ax.set_title(f"{scatter_y_axis} vs {scatter_x_axis}")
        st.pyplot(fig)
    else:
        st.warning("Both X-axis and Y-axis need to be numeric for a scatter plot.")

    # Download filtered data
    st.subheader("Download Processed Data")
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )
