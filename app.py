import os
import io
import json
import traceback
from flask import Flask, request, jsonify
from pypdf import PdfReader
from typing import Dict, Any

# LangChain y Google
import vertexai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_firestore import FirestoreVectorStore
# --- CAMBIO DE LIBRERÍA: Se importa la nueva clase de Embeddings y Chat ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google.cloud import firestore, storage, tasks_v2

# --- Creación de la Aplicación Flask ---
app = Flask(__name__)

# --- Declaración de Clientes Globales ---
clients = {}

def get_clients():
    """
    Inicializa los clientes de Google Cloud de forma 'perezosa' (solo una vez).
    """
    global clients
    if 'firestore' not in clients:
        print("--- Inicializando clientes de Google Cloud por primera vez... ---")
        try:
            env = os.environ
            PROJECT_ID = env.get("PROJECT_ID")
            VERTEX_AI_LOCATION = env.get("VERTEX_AI_LOCATION")
            MODEL_NAME = env.get("GEMINI_MODEL", "gemini-2.5-flash")
            print(f"--- Usando el modelo Gemini: {MODEL_NAME} ---")

            TASK_QUEUE_ID = env.get("TASK_QUEUE_ID") 
            CLOUD_RUN_URL = env.get("CLOUD_RUN_URL")
            
            # La inicialización de vertexai ya no es necesaria para estas librerías,
            # pero la dejamos por si se usan otros servicios de Vertex AI.
            vertexai.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)

            clients['firestore'] = firestore.Client()
            clients['storage'] = storage.Client()
            
            # --- SOLUCIÓN: Se usa la nueva clase GoogleGenerativeAIEmbeddings ---
            # Esta clase acepta el parámetro 'output_dimensionality' directamente.
            clients['embedding'] = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                output_dimensionality=768
            )
            
            # También usamos la clase de chat correspondiente de la nueva librería.
            clients['llm'] = ChatGoogleGenerativeAI(model=MODEL_NAME)
            
            clients['tasks'] = tasks_v2.CloudTasksClient()

            clients['config'] = {
                "project_id": PROJECT_ID,
                "location": VERTEX_AI_LOCATION,
                "queue_id": TASK_QUEUE_ID,
                "run_url": CLOUD_RUN_URL
            }
            
            print("--- Clientes de Google Cloud inicializados correctamente con 'langchain-google-genai'. ---")
        except Exception as e:
            print(f"--- !!! ERROR CRÍTICO durante la inicialización de clientes: {e} ---")
            traceback.print_exc()
            clients = {}
    return clients

# ==============================================================================
# ENDPOINT 1: RECEPTOR DE GCS Y CREADOR DE TAREAS
# ==============================================================================
@app.route("/", methods=["POST"])
def gcs_event_receiver():
    """
    Recibe la notificación de GCS, crea una tarea en Cloud Tasks y responde OK inmediatamente.
    """
    try:
        clients = get_clients()
        tasks_client = clients.get('tasks')
        config = clients.get('config')

        event = request.get_json(silent=True)
        if not event or "bucket" not in event or "name" not in event:
            print("Petición ignorada: evento no válido.")
            return "Evento no válido", 204

        payload = {"bucket": event["bucket"], "filename": event["name"]}
        
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{config['run_url']}/process-task",
                "headers": {"Content-type": "application/json"},
                "body": json.dumps(payload).encode(),
            }
        }

        parent = tasks_client.queue_path(config['project_id'], config['location'], config['queue_id'])
        tasks_client.create_task(parent=parent, task=task)
        
        print(f"Tarea creada para el archivo: {event['name']}")
        return "Tarea creada con éxito", 200

    except Exception as e:
        print(f"!!! ERROR en gcs_event_receiver: {e}")
        traceback.print_exc()
        return "Error al crear la tarea, pero evento aceptado.", 200

# ==============================================================================
# ENDPOINT 2: TRABAJADOR DE PROCESAMIENTO DE PDF
# ==============================================================================
@app.route("/process-task", methods=["POST"])
def pdf_processing_worker():
    """
    Este endpoint es llamado por Cloud Tasks para hacer el trabajo pesado.
    """
    try:
        clients = get_clients()
        storage_client = clients.get('storage')
        if not storage_client:
            return "Error interno: cliente de storage no inicializado", 500

        task_payload = request.get_json(silent=True)
        if not task_payload:
            return "Cuerpo de la petición inválido", 400

        bucket_name = task_payload.get("bucket")
        file_id = task_payload.get("filename")
        
        print(f"Trabajador iniciando procesamiento para: {file_id}")

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_id)
        
        if not blob.exists():
            print(f"El archivo {file_id} no se encontró en el bucket {bucket_name}.")
            return "Archivo no encontrado", 204

        file_bytes = blob.download_as_bytes()
        result = _process_and_embed_pdf_content(file_bytes, file_id)

        if result.get("status") == "ok":
            return "Procesado con éxito por el trabajador", 200
        else:
            reason = result.get("reason", "Razón desconocida")
            return f"Fallo en el procesamiento: {reason}", 500

    except Exception as e:
        print(f"!!! ERROR en pdf_processing_worker: {e}")
        traceback.print_exc()
        return f"Error inesperado en el trabajador: {str(e)}", 500

# ==============================================================================
# FUNCIÓN DE PROCESAMIENTO DE PDF
# ==============================================================================
def _process_and_embed_pdf_content(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Procesa el contenido de un PDF, lo divide, crea embeddings y lo guarda en Firestore.
    """
    try:
        print(f"Iniciando procesamiento para el archivo: {filename}")
        clients = get_clients()
        firestore_client = clients.get('firestore')
        embedding_model = clients.get('embedding')

        if not firestore_client or not embedding_model:
            raise Exception("Los clientes de Firestore o Embedding no se inicializaron correctamente.")

        print("Paso 1/4: Leyendo contenido del PDF...")
        reader = PdfReader(io.BytesIO(file_bytes))
        text_content = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not text_content:
            return {"status": "error", "reason": "No se pudo extraer texto del PDF."}

        print("Paso 2/4: Dividiendo el texto en fragmentos...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = text_splitter.split_text(text_content)
        
        book_title = os.path.splitext(filename)[0].replace("_", " ").title()
        
        print(f"Paso 3/4: Creando {len(chunks)} objetos Document enriquecidos...")
        documents = [Document(page_content=f"Título del documento: {book_title}. Contenido: {chunk}", metadata={"source": filename, "chunk_index": i, "title": book_title}) for i, chunk in enumerate(chunks)]
        
        COLLECTION_NAME = "pdf_embeded_documents"
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME,
            embedding_service=embedding_model,
            client=firestore_client,
        )

        batch_size = 100 
        print(f"Paso 4/4: Guardando {len(documents)} documentos en Firestore en lotes de {batch_size}...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"  -> Guardando lote {int(i/batch_size) + 1} de {int(len(documents)/batch_size) + 1}...")
            vector_store.add_documents(batch)
        
        print(f"Procesamiento completado con éxito para: {filename}")
        return {"status": "ok", "message": f"Archivo {filename} procesado."}

    except Exception as e:
        print(f"!!! ERROR dentro de _process_and_embed_pdf_content para {filename}: {e}")
        traceback.print_exc()
        return {"status": "error", "reason": str(e)}

# ==============================================================================
# ENDPOINT DE CONSULTA RAG
# ==============================================================================
@app.route("/query", methods=["POST"])
def query_rag_handler():
    try:
        request_data = request.get_json()
        if not request_data or "query" not in request_data:
            return jsonify({"error": "Petición inválida. Se requiere 'query'."}), 400
        
        user_query = request_data["query"]

        clients = get_clients()
        firestore_client = clients.get('firestore')
        embedding_model = clients.get('embedding')

        if not firestore_client or not embedding_model:
            return jsonify({"error": "Error interno del servidor."}), 500

        COLLECTION_NAME = "pdf_embeded_documents"
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME,
            embedding_service=embedding_model,
            client=firestore_client,
        )
        
        found_docs = vector_store.similarity_search(query=user_query, k=4)

        results = [{"source": doc.metadata.get("source", "N/A"), "content": doc.page_content} for doc in found_docs]
        
        return jsonify({"results": results}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error inesperado: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
