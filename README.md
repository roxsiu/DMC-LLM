# DMC-LLM
Proyecto del Grupo 4 del Curso LLM

# Proyecto RAG para Pólizas (.txt) — Streamlit + LangChain + FAISS

Este repositorio implementa un asistente RAG (Retrieval‑Augmented Generation) para analizar pólizas de seguro en formato `.txt`. Permite indexar el contenido y hacer preguntas con una interfaz en Streamlit.

---

## Requisitos

* Python 3.9 o superior
* GitHub Codespaces (recomendado) o entorno local con Python




3. Instalar dependencias (dos opciones):

   * Con `requirements.txt`:

     ```bash
     pip install -r requirements.txt
     ```
   * O instalación directa:

     ```bash
     pip install streamlit==1.39.0 \
                 langchain==0.3.7 \
                 langchain-community==0.3.7 \
                 langchain-text-splitters==0.3.2 \
                 faiss-cpu==1.8.0.post1 \
                 sentence-transformers==2.7.0 \
                 tiktoken==0.7.0 \
                 openai==1.51.2 \
                 python-dotenv==1.0.1
     ``


## Estructura del repositorio

```
<repo_root>/
├─ RAG4.py        # Aplicación principal 
├─ requirements.txt         # Lista de dependencias 
├─ .env                     # Clave API (NO subir a GitHub)
├─ .gitignore               # Debe incluir .env
└─ README.md                # Este documento
