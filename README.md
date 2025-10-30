# Proyecto RAG - Consulta de Pólizas de Seguro

Aplicación en **Streamlit** que permite consultar el contenido de una póliza de seguros en texto plano, utilizando un enfoque de **RAG (Retrieval-Augmented Generation)**.  
El usuario puede subir un archivo `.txt` y realizar preguntas sobre su contenido.

---

## Descripción técnica

El proyecto implementa un flujo RAG básico:

1. **Carga del documento (.txt)**  
   El usuario sube una póliza en texto plano.

2. **División en fragmentos (chunking)**  
   Se usa `RecursiveCharacterTextSplitter` para dividir el texto en fragmentos de 1000 caracteres con 150 de solapamiento.

3. **Generación de embeddings**  
   Se crean vectores numéricos mediante `OpenAIEmbeddings` (modelo `text-embedding-3-small`).

4. **Almacenamiento vectorial y búsqueda**  
   Los embeddings se guardan en una base FAISS para búsquedas de similitud.

5. **Generación de respuesta**  
   Los fragmentos más relevantes se envían al modelo `gpt-3.5-turbo`, que genera una respuesta contextualizada.

---

## Tecnologías utilizadas

| Componente | Herramienta |
|-------------|-------------|
| Interfaz | Streamlit |
| Framework | LangChain |
| LLM | GPT-3.5-Turbo |
| Embeddings | OpenAIEmbeddings |
| Vector Store | FAISS |
| Splitter | RecursiveCharacterTextSplitter |
| Variables de entorno | python-dotenv |


