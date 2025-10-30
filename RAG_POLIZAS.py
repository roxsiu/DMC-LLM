# ---------------------------------------------------------------------
# 🔹 IMPORTS
# ---------------------------------------------------------------------
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ---------------------------------------------------------------------
# 🔹 Cargar la API Key desde el archivo .env
# ---------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ No se encontró la variable OPENAI_API_KEY. Usa 'echo \"OPENAI_API_KEY=tu_api_key\" >> .env'")
else:
    os.environ["OPENAI_API_KEY"] = api_key

# ---------------------------------------------------------------------
# 🔹 Configuración de la app Streamlit
# ---------------------------------------------------------------------
st.set_page_config(page_title="RAG de Pólizas", layout="centered")
st.title("Asistente para consultar pólizas de seguro en texto plano")

st.sidebar.header("💡 Sugerencias de preguntas")
st.sidebar.markdown("""
- ¿Cuál es el deducible aplicable?  
- ¿Qué coberturas incluye la póliza?  
- ¿Qué hacer en caso de siniestro?
""")

# ---------------------------------------------------------------------
# 🔹 Subida del archivo
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Sube el archivo de póliza (.txt)", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    with st.spinner("Construyendo el índice vectorial... Esto puede tardar unos segundos."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(text)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # 🔹 Crear base vectorial en memoria con Chroma
        vectorstore = Chroma.from_texts(
            chunks,
            embeddings,
            collection_name="polizas_temp"
        )

        # 🔹 Crear el modelo LLM y el RAG
        llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-4o-mini",
            openai_api_key=api_key
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    st.success("✅ Índice vectorial creado correctamente. Ya puedes hacer preguntas.")

    # Campo de entrada de pregunta (solo después del índice)
    user_query = st.text_input("Escribe tu pregunta:")
    if user_query:
        with st.spinner("Buscando respuesta..."):
            result = qa_chain(user_query)

        st.markdown("### 🧠 Respuesta:")
        st.markdown(result["result"])

        if "source_documents" in result:
            st.markdown("### 📄 Fragmento utilizado para la respuesta:")
            st.info(result["source_documents"][0].page_content)

else:
    st.info("📄 Por favor, sube un archivo de texto (.txt) para comenzar.")
