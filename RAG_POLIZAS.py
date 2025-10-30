# App RAG para pólizas 
# ---------------------------------------------------------------
# Cambios mínimos:
# - Acepta .txt
# - Usa CharacterTextSplitter + SentenceTransformerEmbeddings + FAISS
# - Usa RetrievalQA (patrón clásico) con ChatOpenAI si hay API key

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI  # ✅ IMPORT CORRECTO PARA CHAT
from langchain.chains import RetrievalQA

# ---------------------------------------------------------------------
# 🔹 Cargar la API key
# ---------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la variable OPENAI_API_KEY. Usa 'echo \"OPENAI_API_KEY=tu_api_key\" >> .env'")
else:
    os.environ["OPENAI_API_KEY"] = api_key

# ---------------------------------------------------------------------
# 🔹 Configuración de Streamlit
# ---------------------------------------------------------------------
st.set_page_config(page_title="RAG de Pólizas", layout="centered")
st.title("Asistente para consultar pólizas de seguro en texto plano")

st.sidebar.header("💡 Sugerencias de preguntas")
st.sidebar.markdown("""
- ¿Qué cubre la póliza?
- ¿Cuál es el deducible aplicable?
- ¿Qué hacer en caso de siniestro?
""")

# ---------------------------------------------------------------------
# 🔹 Subida de archivo y pregunta
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Sube el archivo de póliza (.txt)", type=["txt"])
user_query = st.text_input("Escribe tu pregunta:")
search_button = st.button("Buscar")

# ---------------------------------------------------------------------
# 🔹 Procesamiento del documento
# ---------------------------------------------------------------------
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    with st.spinner("Construyendo el índice vectorial..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # ✅ Usamos modelo de chat moderno
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo"
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

    st.success("✅ Índice vectorial construido correctamente.")

    # -----------------------------------------------------------------
    # 🔹 Responder preguntas
    # -----------------------------------------------------------------
    if search_button and user_query:
        with st.spinner("Buscando respuesta..."):
            result = qa_chain.run(user_query)
        st.markdown(f"**Respuesta:** {result}")

else:
    st.info("📄 Por favor, sube un archivo de texto (.txt) para comenzar.")