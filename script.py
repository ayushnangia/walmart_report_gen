import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from fpdf import FPDF
import pandas as pd
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import subprocess


DB_FAISS_PATH = 'vectorstore/db_faiss'
import os  
def dataframe_to_image(df):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(df.shape[1], df.shape[0])) 
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    return img

def generate_pdf_report(file_names, file_paths, chat_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(10, 10, 10)  
    pdf.set_font("Arial", 'B', 16)


    pdf.cell(0, 10, "Sales Forecasting Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    for file_name, file_path in zip(file_names, file_paths):
        pdf.cell(0, 10, f"Data from {file_name}", ln=True, align='L')
        pdf.ln(5)  

        df = pd.read_csv(file_path)

        if len(df.columns) > 3:
            img = dataframe_to_image(df.head(5))  
            img_path = f"temp_{uuid.uuid4()}.png"
            img.save(img_path)
            pdf.image(img_path, x=10, y=None, w=180)  
            os.remove(img_path)  
        else:
            base_col_width = pdf.w / 4
            for col in df.columns:
                pdf.cell(base_col_width, 10, col, border=1, align='C')
            pdf.ln()  
            for i in range(min(5, len(df))):
                for val in df.iloc[i].values:
                    pdf.cell(base_col_width, 10, str(val), border=1, align='C')
                pdf.ln(10)  
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Chat History:", ln=True, align='L')
    pdf.set_font("Arial", size=12)
    for query, response in chat_history:
        pdf.multi_cell(0, 10, f"Q: {query}", align='L')
        pdf.multi_cell(0, 10, f"A: {response}", align='L')
        pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Exploratory data analysis:", ln=True, align='L')
    pdf.set_font("Arial", size=12)

    plot_dir = 'plot'  
    if os.path.exists(plot_dir):
        for img_file in os.listdir(plot_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(plot_dir, img_file)
                pdf.image(img_path, x=10, y=None, w=180)  
                pdf.ln(1)  

                pdf.set_font("Arial", size=10)
                pdf.cell(0, 10, img_file, ln=True, align='C')  
                pdf.ln(10)


    script_path = 'p4.py'  
    subprocess.run(['python', script_path], check=True)
    results_csv_path = 'output/model_comparison_results.csv' 
    if os.path.exists(results_csv_path):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Results:", ln=True, align='L')
        pdf.set_font("Arial", size=12)
        pdf.ln(5)

        results_df = pd.read_csv(results_csv_path)
        num_columns = len(results_df.columns) - 1  
        base_col_width = (pdf.w - 20) / num_columns  # Adjust column width based on number of columns

        for col in results_df.columns[1:]:
            pdf.cell(base_col_width, 10, col, border=1, align='C')
        pdf.ln()

        for _, row in results_df.iterrows():
            for val in row[1:]:  
                pdf.cell(base_col_width, 10, str(val), border=1, align='C')
            pdf.ln()


    unique_id = str(uuid.uuid4())
    pdf_output = f'data/report_{unique_id}.pdf'
    pdf.output(pdf_output)
    return pdf_output





#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

st.title("Sales Forcasting - Data analysis with chatbot and report generation")
uploaded_files = st.sidebar.file_uploader("Upload your Data", type="csv",accept_multiple_files=True)

if uploaded_files :
    uploaded_file=uploaded_files[0]
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                'delimiter': ','})
    data = loader.load()
    #st.json(data)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

    if len(uploaded_files) > 1:
        if st.sidebar.button("Generate Report"):
            tmp_file_paths = []
            file_names = [file.name for file in uploaded_files]  # Original file names
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_paths.append(tmp_file.name)

            report_path = generate_pdf_report(file_names, tmp_file_paths, st.session_state['history'])
            with open(report_path, "rb") as file:
                st.sidebar.download_button(
                    label="Download Report",
                    data=file,
                    file_name="report.pdf",
                    mime="application/octet-stream"
                )


