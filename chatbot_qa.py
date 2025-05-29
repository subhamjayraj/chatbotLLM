import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import IAMTokenManager
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os


API_KEY = os.getenv("WATSONX_API_KEY")
ENDPOINT = os.getenv("WATSONX_ENDPOINT")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# Streamlit UI
st.header("WatsonX PDF Chatbot")

with st.sidebar:
    st.title("Upload your PDF")
    file = st.file_uploader("Upload a PDF file and ask questions", type="pdf")

# Extract text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n", ".", "!", "?", " "],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # Embedding - use a HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

    vector_store = st.session_state.vector_store

    # Get user input
    user_question = st.text_input("Ask a question")

    if user_question:
        docs = vector_store.similarity_search(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful assistant. Use the following document excerpts to answer the question.

Document:
{context}

Question:
{user_question}

Answer:"""

        # WatsonX token manager and model
        token_manager = IAMTokenManager(api_key=API_KEY)
        model = ModelInference(
            model_id="google/flan-ul2",
            project_id=PROJECT_ID,
            url=ENDPOINT,
            token_manager=token_manager
        )

        parameters = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 300,
            GenParams.TEMPERATURE: 0.2
        }

        response = model.generate_text(prompt=prompt, params=parameters)
        st.write(response["results"][0]["generated_text"])
