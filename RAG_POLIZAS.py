import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# ------------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ------------------------------------------------------------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("No se encontró la variable OPENAI_API_KEY. Agrega tu clave al archivo .env.")
    st.stop()

# ------------------------------------------------------------
# CONFIGURACIÓN DE LA APP
# ------------------------------------------------------------

st.set_page_config(page_title="Asistente RAG de Pólizas", layout="centered")
st.title("Asistente RAG de Pólizas de Seguros")
st.markdown("Sube un archivo de texto con el contenido de la póliza y realiza consultas sobre ella.")

# Sidebar con sugerencias
st.sidebar.header("Sugerencias de preguntas")
st.sidebar.markdown("""
- ¿Qué coberturas incluye la póliza?  
- ¿Cuál es el deducible aplicable?  
- ¿Qué debo hacer en caso de siniestro?  
- ¿Cuándo inicia la cobertura?  
- ¿Cuáles son las exclusiones principales?
""")

# ------------------------------------------------------------
# SUBIDA Y PROCESAMIENTO DEL ARCHIVO
# ------------------------------------------------------------

uploaded_file = st.file_uploader("📄 Sube el archivo de póliza (.txt)", type="txt")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    st.success(f"Índice vectorial creado con {len(chunks)} fragmentos.")

    # ------------------------------------------------------------
    # CREAR ÍNDICE VECTORIAL CON CHROMA (LangChain)
    # ------------------------------------------------------------

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="polizas_temp",
        persist_directory=None  # evita escribir archivos en Cloud
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ------------------------------------------------------------
    # CONFIGURAR MODELO Y PROMPT
    # ------------------------------------------------------------

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.3
    )

    prompt_template = """
    Eres un asistente experto en pólizas de seguros.
    Usa exclusivamente la información del contexto para responder de forma clara y breve.
    Si la información no está en el contexto, responde: "No tengo información suficiente en la póliza para responder eso."

    Contexto:
    {context}

    Pregunta:
    {question}
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # ------------------------------------------------------------
    # INTERFAZ DE PREGUNTAS Y RESPUESTAS
    # ------------------------------------------------------------

    question = st.text_input("Escribe tu pregunta sobre la póliza:")

    if question:
        result = qa_chain({"query": question})

        st.markdown("###  Respuesta:")
        st.write(result["result"])

        if result.get("source_documents"):
            st.markdown("### 📄 Fragmento más relevante utilizado:")
            st.write(result["source_documents"][0].page_content.strip())

else:
    st.info("Por favor, sube un archivo de texto para comenzar.")
