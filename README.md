# Walmart Sales and Data Analysis with Chatbot Interface

Welcome to the Walmart Sales and Data Analysis project, an innovative solution that combines the power of machine learning, natural language processing, and data visualization to provide insightful sales data analysis. This project features a conversational chatbot interface built with Streamlit, allowing for an interactive exploration of sales data and automated report generation.

## Overview

This tool is designed to simplify the analysis of sales data by providing a user-friendly interface where users can upload their sales data, ask questions in natural language, and receive insights and forecasts. It leverages the Llama 2 7B Chat model for understanding and processing user queries, ensuring accurate and relevant responses. The tool also features an automated process for generating comprehensive PDF reports that include data summaries, exploratory data analysis (EDA) results, chat history, and forecasting insights.

## Features

- **Interactive Data Upload:** Supports uploading of CSV files containing sales data.
- **Conversational Chatbot Interface:** Powered by the Llama 2 7B Chat model for natural language query processing.
- **Exploratory Data Analysis (EDA):** Automated visualization and summary statistics generation.
- **Sales Forecasting:** Advanced machine learning models for accurate sales forecasting.
- **Automated PDF Report Generation:** Summarizes analysis findings, chat interactions, and forecasting results.

## Technical Stack

- **Streamlit:** For the interactive web application.
- **LangChain:** Powers the conversational queries and integrates the chatbot with data.
- **HuggingFace & FAISS:** Manages embeddings and efficient data retrieval.
- **FPDF:** Generates PDF reports.
- **Matplotlib & Seaborn:** Creates visualizations.
- **Pandas:** Handles data manipulation and analysis.

## Integrating Llama 2 7B Chat Model

1. **Download the Model:**
   - Visit [Llama-2-7B-Chat-GGML on Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML).
   - Download the `llama-2-7b-chat.ggmlv3.q8_0.bin` model file.

2. **Prepare the Model Directory:**
   ```
   mkdir -p model
   ```

3. **Place the Model File:**
   ```
   mv /path/to/llama-2-7b-chat.ggmlv3.q8_0.bin ./model/
   ```

## Getting Started

### Setup Environment

1. **Clone the Repository:**
   ```
   git clone https://github.com/ayushnangia/walmart_report_gen
   ```

2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```

### Run the Application

```
streamlit run script.py
```

### Use the Application

- Upload sales data CSV files using the interface.
- Interact with the chatbot to explore the data.
- Generate and download comprehensive PDF reports.

