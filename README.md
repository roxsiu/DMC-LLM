# RAG de Pólizas de Seguro

Aplicación **Streamlit** que permite consultar pólizas de seguro en texto plano utilizando un enfoque de **Retrieval-Augmented Generation (RAG)**.

---

## Descripción

La aplicación:
- Permite subir un archivo `.txt` con el contenido de una póliza.
- Divide el texto en fragmentos (`RecursiveCharacterTextSplitter`).
- Usa **OpenAI Embeddings** y **Chroma** para crear un índice vectorial local.
- Recupera el fragmento más relevante para responder a la pregunta del usuario.
- Muestra la **respuesta generada** y el **fragmento utilizado**.

---

## Requisitos

Asegúrate de tener instalado **Python 3.12** o superior.

### Librerías principales
Las dependencias se encuentran en `requirements.txt`:

