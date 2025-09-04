# app.py

import os
import traceback
from flask import Flask, request
from pypdf import PdfReader
from typing import List, Dict, Any

# LangChain y Google
import vertexai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from google.cloud import firestore, storage

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
            PROJECT_ID = os.environ.get("PROJECT_ID")
            VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION")
            
            vertexai.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)

            clients['firestore'] = firestore.Client()
            clients['storage'] = storage.Client()
            clients['embedding'] = VertexAIEmbeddings(model_name="text-embedding-004")
            # Nota: El modelo se corrigió a uno válido, pero usa el que necesites.
            clients['llm'] = ChatVertexAI(model_name="gemini-1.5-flash-001")
            
            print("--- Clientes de Google Cloud inicializados correctamente. ---")
        except Exception as e:
            print(f"--- !!! ERROR CRÍTICO durante la inicialización de clientes: {e} ---")
            traceback.print_exc()
            clients = {} # Resetea en caso de fallo para reintentar
    return clients

# ==============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO DE PDF
# ==============================================================================

def _process_and_embed_pdf_content(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Procesa el contenido de un PDF, lo divide, crea embeddings y lo guarda en Firestore.
    Esta función ahora maneja errores internamente y siempre devuelve un diccionario.
    """
    try:
        print(f"Iniciando procesamiento para el archivo: {filename}")
        clients = get_clients()
        firestore_client = clients.get('firestore')
        embedding_model = clients.get('embedding')

        if not firestore_client or not embedding_model:
            raise Exception("Los clientes de Firestore o Embedding no se inicializaron correctamente.")

        # ---
        # --- TODO: AQUÍ VA TU LÓGICA PARA PROCESAR EL PDF ---
        # Este es un ejemplo de cómo podría ser la lógica. Adáptalo a tus necesidades.
        # ---

        # 1. Leer el contenido del PDF
        # reader = PdfReader(io.BytesIO(file_bytes))
        # text_content = "".join(page.extract_text() for page in reader.pages)
        # if not text_content:
        #     return {"status": "error", "reason": "No se pudo extraer texto del PDF."}

        # 2. Dividir el texto en chunks
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        # chunks = text_splitter.split_text(text_content)
        
        # 3. Crear objetos Document de LangChain
        # documents = [
        #     Document(
        #         page_content=chunk,
        #         metadata={"source": filename, "chunk_index": i}
        #     ) for i, chunk in enumerate(chunks)
        # ]
        
        # 4. Inicializar y usar el Vector Store para añadir los documentos
        # COLLECTION_NAME = "mi_coleccion_de_vectores" # Reemplaza con tu nombre de colección
        # vector_store = FirestoreVectorStore(
        #     collection=COLLECTION_NAME,
        #     embedding_service=embedding_model,
        #     client=firestore_client,
        # )
        # vector_store.add_documents(documents)
        
        # --- Fin de la sección TODO ---
        
        print(f"Procesamiento completado con éxito para: {filename}")
        return {"status": "ok", "message": f"Archivo {filename} procesado y añadido al vector store."}

    except Exception as e:
        # Si algo falla en cualquier punto, se captura el error aquí.
        print(f"!!! ERROR dentro de _process_and_embed_pdf_content para {filename}: {e}")
        traceback.print_exc()  # Imprime el error detallado en los logs para depuración
        return {"status": "error", "reason": str(e)}

# ==============================================================================
# ENDPOINT PRINCIPAL AUTOMATIZADO
# ==============================================================================
@app.route("/", methods=["POST"])
def handle_gcs_event():
    try:
        clients = get_clients()
        storage_client = clients.get('storage')

        if not storage_client:
            print("Error: Storage client no inicializado.")
            return "Error interno del servidor", 500

        event = request.get_json(silent=True)
        if not event:
            print("Petición POST recibida sin cuerpo JSON. Ignorando.")
            return "Petición ignorada", 204

        bucket_name = event.get("bucket")
        file_id = event.get("name")
        
        if not bucket_name or not file_id:
            print(f"Evento ignorado: no es un evento de Cloud Storage válido. Evento: {event}")
            return "Evento no válido", 204

        print(f"Archivo detectado: {file_id} en bucket: {bucket_name}")

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_id)
        if not blob.exists():
            print(f"Archivo {file_id} no encontrado en el bucket.")
            return "Archivo no encontrado para procesar", 204
            
        file_bytes = blob.download_as_bytes()
        
        # Esta llamada ahora es segura porque la función siempre devuelve un diccionario
        result = _process_and_embed_pdf_content(file_bytes, file_id)
        
        # Esta comprobación ya no causará un error
        if result.get("status") == "ok":
            print(f"Éxito al procesar {file_id}.")
            return "Procesado con éxito", 200
        else:
            reason = result.get("reason", "Razón desconocida")
            print(f"Fallo al procesar {file_id}: {reason}")
            # Devolvemos 200 para que GCS no reintente el evento, pero logueamos el error.
            return f"Fallo en el procesamiento: {reason}", 200

    except Exception as e:
        print(f"Error inesperado procesando el evento: {traceback.format_exc()}")
        return f"Error inesperado: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
