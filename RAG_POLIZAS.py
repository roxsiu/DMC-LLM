# ---------------------------------------------------------------
# RAG_POLIZAS.py - Asistente para consultar pólizas de seguro
# Versión: Fragmento ganador con score (FAISS + OpenAI)
# ---------------------------------------------------------------

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import openai

# ---------------------------------------------------------------
# Cargar clave de API
# ---------------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("No se encontró la variable OPENAI_API_KEY. Usa 'echo \"OPENAI_API_KEY=tu_api_key\" >> .env'")
else:
    openai.api_key = api_key

# ---------------------------------------------------------------
# Configuración general de Streamlit
# ---------------------------------------------------------------
st.set_page_config(page_title="RAG de Pólizas", layout="centered")
st.title("Asistente para consultar pólizas de seguro")

st.sidebar.header("Ejemplos de preguntas")
st.sidebar.markdown("""
- ¿Qué cubre la póliza?
- ¿Cuál es el deducible aplicable?
- ¿Qué hacer en caso de siniestro?
- ¿Cuándo vence la cobertura?
""")

# ---------------------------------------------------------------
# Carga del archivo de texto
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Sube el archivo de póliza (.txt)", type=["txt"])

# ---------------------------------------------------------------
# Procesamiento del documento
# ---------------------------------------------------------------
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

    with st.spinner("Procesando la póliza..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_text(text)

        # Crear embeddings y vectorstore con FAISS
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)

    # Mostrar confirmación antes del campo de pregunta
    st.success("Índice vectorial creado correctamente.")

    # Campo de pregunta
    user_query = st.text_input("Escribe tu pregunta:")
    search_button = st.button("Buscar")

    # -----------------------------------------------------------
    # Consulta y respuesta
    # -----------------------------------------------------------
    if search_button and user_query:
        with st.spinner("Buscando respuesta..."):
            # Recuperar varios candidatos con sus scores
            results = vectorstore.similarity_search_with_score(user_query, k=5)

            # Elegir el fragmento con menor score (más relevante)
            best_doc, best_score = min(results, key=lambda x: x[1])
            context = best_doc.page_content

            # Prompt anclado al contexto ganador
            prompt = f"""Responde SOLO usando el siguiente contexto de la póliza.
Si la respuesta no está en el contexto, di: "No se encuentra en la póliza".

Contexto:
\"\"\"{context}\"\"\"

Pregunta: {user_query}

Respuesta:"""

            # Llamada al modelo instruct
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=400,
                temperature=0.2
            )

            answer = response.choices[0].text.strip()

        # -------------------------------------------------------
        # Mostrar resultados
        # -------------------------------------------------------
        st.markdown("### Respuesta:")
        st.markdown(answer)

        st.markdown("---")
        st.markdown("#### Fragmento utilizado para la respuesta:")
        st.markdown(best_doc.page_content)
        st.caption(f"Score de similitud (distancia FAISS): {best_score:.4f}")

        # Mostrar los 3 mejores candidatos (opcional)
        with st.expander("Ver otros fragmentos candidatos"):
            for rank, (doc, score) in enumerate(sorted(results, key=lambda x: x[1])[:3], start=1):
                st.markdown(f"**Candidato {rank} — score: {score:.4f}**")
                st.write(doc.page_content)
                st.markdown("---")

else:
    st.info("Por favor, sube un archivo de texto para comenzar.")