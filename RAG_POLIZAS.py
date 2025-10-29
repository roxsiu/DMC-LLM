# App RAG para pólizas 
# ---------------------------------------------------------------
# Cambios mínimos:
# - Acepta .txt
# - Usa CharacterTextSplitter + SentenceTransformerEmbeddings + FAISS
# - Usa RetrievalQA (patrón clásico) con ChatOpenAI si hay API key


import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# -------------------------------------------------------------
# CARGA DE VARIABLES Y CONFIGURACIÓN
# -------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Pólizas", layout="wide")
st.title("RAG PÓLIZAS (.txt)")
st.write("Asistente para consultar pólizas de seguro en formato texto plano.")

# -------------------------------------------------------------
# SIDEBAR – PREGUNTAS SUGERIDAS
# -------------------------------------------------------------
st.sidebar.header("Preguntas sugeridas")

suggested_questions = [
    "¿Qué cubre esta póliza?",
    "¿Qué exclusiones menciona el documento?",
    "¿Cuál es el deducible aplicable?"
]

selected_question = st.sidebar.radio(
    "Selecciona una pregunta o escribe la tuya:",
    suggested_questions
)

# -------------------------------------------------------------
# FUNCIONES AUXILIARES
# -------------------------------------------------------------
def split_by_titles(text):
    """Divide el texto por títulos en mayúsculas."""
    sections = re.split(r"\n(?=[A-ZÁÉÍÓÚÑ¿]{3,}.*\n)", text)
    cleaned_sections = [s.strip() for s in sections if len(s.strip()) > 0]
    return cleaned_sections

def chunk_text(text):
    """Combina secciones en mayúsculas y división por tamaño."""
    sections = split_by_titles(text)
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    for sec in sections:
        chunks.extend(splitter.split_text(sec))
    return chunks

def build_vectorstore(file_path):
    """Crea el índice vectorial FAISS a partir del texto."""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text = "\n".join([d.page_content for d in documents])
    chunks = chunk_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embeddings)
    return db

def get_llm():
    """Carga el modelo LLM de OpenAI si hay API key."""
    if api_key:
        return ChatOpenAI(temperature=0.2, model="gpt-4o-mini", openai_api_key=api_key)
    else:
        st.warning("No se detectó OPENAI_API_KEY. Se usará solo búsqueda semántica.")
        return None

def create_qa_chain(vectorstore, llm):
    """Crea la cadena RAG (Retriever + Modelo)."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    if llm:
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    else:
        chain = retriever
    return chain

# -------------------------------------------------------------
# INTERFAZ STREAMLIT
# -------------------------------------------------------------
uploaded_file = st.file_uploader("Sube el archivo de póliza (.txt)", type=["txt"])

if uploaded_file is not None:
    temp_path = "temp_policy.txt"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info("Construyendo el índice vectorial... Esto puede tardar unos segundos.")
    db = build_vectorstore(temp_path)
    llm = get_llm()
    qa_chain = create_qa_chain(db, llm)

    st.success("Índice creado correctamente. Ya puedes hacer preguntas sobre la póliza.")

    # Entrada de texto principal
    query = st.text_input("Escribe tu pregunta:", value=selected_question)

    if query:
        if llm:
            response = qa_chain.run(query)
            st.write("**Respuesta:**", response)
        else:
            docs = db.similarity_search(query, k=3)
            st.write("**Textos más relevantes:**")
            for i, doc in enumerate(docs, 1):
                st.write(f"**{i}.** {doc.page_content[:400]}...")
else:
    st.info("Sube un archivo .txt de una póliza para comenzar.")