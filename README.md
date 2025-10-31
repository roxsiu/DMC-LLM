# DMC-LLM
Proyecto del Grupo 4 del Curso LLM:
Roxana Siu 
Juan Carlos Ramos
Fernando Gilardi

# Asistente RAG para Pólizas de Seguro

Aplicación Streamlit que permite cargar pólizas en texto plano (`.txt`) y consultar su contenido mediante lenguaje natural, usando un enfoque RAG (Retrieval-Augmented Generation) con OpenAI y Chroma como base vectorial.

Versión en la nube: https://dmc-llm-polizas.streamlit.app/

---

## Cómo ejecutar la app (local)

### 1) Clonar el repositorio

git clone https://github.com/roxsiu/DMC-LLM.git
cd DMC-LLM

### 2) Crear entorno virtual

python -m venv .venv

#### Linux/Mac:
source .venv/bin/activate

#### Windows:
.venv\Scripts\activate

### 3) Instalar dependencias

pip install -r requirements.txt

### 4) Configurar la clave de OpenAI

OPENAI_API_KEY=tu_api_key_aqui

Asegúrate de que .env está en .gitignore para no subirlo a GitHub.

### 5) Ejecutar la aplicación

streamlit run RAG_POLIZAS.py

### 6) Descripción técnica


#### Carga del documento
El usuario sube un archivo `.txt` con el contenido de una póliza.

#### Procesamiento
El texto se divide en fragmentos con `RecursiveCharacterTextSplitter`.  
Cada fragmento se convierte en un *embedding* y se guarda en **ChromaDB**.

#### Consulta
El usuario formula una pregunta en lenguaje natural.  
Se recuperan los fragmentos más relevantes y el modelo **GPT** genera la respuesta final.


