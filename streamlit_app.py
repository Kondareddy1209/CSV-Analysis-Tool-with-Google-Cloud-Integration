import sys
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import storage

# Get the file path from command line argument
file_path = sys.argv[1]

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Konda Reddy\Documents\GitHub\gen-lang-client-0298324082-8eef79011259.json"

# Function to upload file to Google Cloud Storage
def upload_to_gcs(bucket_name, file):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file.name)
    blob.upload_from_string(file.read(), content_type="text/csv")
    return f"gs://{bucket_name}/{file.name}"

# Streamlit UI
st.set_page_config(page_title="CSV Analysis", layout="wide")
st.title("CSV Analysis Tool with Google Cloud Integration")
st.write("Upload your CSV file for analysis and storage in Google Cloud Storage.")

# Step 1: Upload to Google Cloud Storage
bucket_name = "cropyieldprediction"  # Replace with your bucket name
with open(file_path, "rb") as file:
    gcs_path = upload_to_gcs(bucket_name, file)
    st.success(f"File successfully uploaded to: {gcs_path}")

# Step 2: Load the file into a Pandas DataFrame
data = pd.read_csv(file_path)
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Step 3: Display basic statistics
st.subheader("Basic Statistics")
st.write(data.describe())

# Step 4: Filtering and Analysis
st.subheader("Filter Data")
filter_col = st.selectbox("Select a column to filter", data.columns)
unique_values = data[filter_col].unique()
selected_values = st.multiselect(f"Select values to filter '{filter_col}':", unique_values)

if selected_values:
    filtered_data = data[data[filter_col].isin(selected_values)]
    st.dataframe(filtered_data)
else:
    filtered_data = data

# Step 5: Data Visualization
st.subheader("Bar Chart")
x_axis = st.selectbox("Select X-axis column for Bar Chart", data.columns)
y_axis = st.selectbox("Select Y-axis column for Bar Chart", data.columns)

if pd.api.types.is_numeric_dtype(filtered_data[y_axis]):
    st.write(f"Visualizing {y_axis} vs {x_axis}")
    chart_data = filtered_data[[x_axis, y_axis]].groupby(x_axis).mean()
    st.bar_chart(chart_data)
else:
    st.warning(f"Cannot visualize non-numeric data in the {y_axis} column.")

# Pie Chart
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

# Line Graph
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

# Scatter Plot
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

# Step 6: Allow downloading of filtered data
st.subheader("Download Processed Data")
if st.button("Download Data"):
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )
