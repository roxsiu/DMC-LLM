# RAG_POLIZAS.py
# -------------------------------------------------------------
# Asistente RAG para pólizas de seguros en texto plano.
# Compatible con Streamlit Cloud (Python 3.11 / 3.12) y OpenAI API.

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# -------------------------------------------------------------
# Cargar API key desde .env
# -------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la variable OPENAI_API_KEY. Asegúrate de definirla en el archivo .env o como secreto en Streamlit Cloud.")
else:
    os.environ["OPENAI_API_KEY"] = api_key

# -------------------------------------------------------------
# Configuración de la app Streamlit
# -------------------------------------------------------------
st.set_page_config(page_title="RAG de Pólizas", layout="centered")
st.title("Asistente para consultar pólizas de seguro")

# Sidebar con sugerencias
st.sidebar.header("Sugerencias de preguntas")
st.sidebar.markdown("""
- ¿Cuál es el deducible aplicable?  
- ¿Qué coberturas incluye la póliza?  
- ¿Qué hacer en caso de siniestro?  
""")

# -------------------------------------------------------------
# Subir archivo de texto
# -------------------------------------------------------------
uploaded_file = st.file_uploader("Sube el archivo de póliza (.txt)", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    with st.spinner("Procesando documento y construyendo el índice vectorial..."):
        # Dividir el texto en fragmentos
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(text)

        # Crear embeddings con OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # Crear base vectorial con Chroma
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,              # ✅ correcto para LangChain 0.0.284
            collection_name="polizas"
        )

        # Crear el modelo de lenguaje
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4o-mini",
            openai_api_key=api_key
        )

        # Crear la cadena de recuperación
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    st.success("✅ Índice vectorial creado correctamente.")

    # Campo para preguntas (después del índice)
    user_query = st.text_input("Escribe tu pregunta:")
    search_button = st.button("Buscar")

    # Procesar la pregunta
    if search_button and user_query:
        with st.spinner("Buscando respuesta..."):
            result = qa_chain(user_query)

        # Mostrar respuesta
        st.markdown("### Respuesta:")
        st.markdown(result["result"])

        # Mostrar fragmento más relevante
        if "source_documents" in result:
            source = result["source_documents"][0].page_content
            st.markdown("### Fragmento utilizado:")
            st.write(source)

else:
    st.info("📄 Sube un archivo de texto (.txt) para comenzar.")

