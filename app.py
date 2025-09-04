# app.py

import os
import hashlib
import io
import traceback
import base64
import json
import re
from datetime import datetime, timezone

from flask import Flask, request
from pypdf import PdfReader
from typing import List, Dict, Any

import vertexai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

app = Flask(__name__)

# --- Declaración de Clientes Globales (sin inicializar) ---
clients = {}

def get_clients():
    """Inicializa los clientes de Google Cloud de forma 'perezosa' (solo una vez)."""
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
            clients['llm'] = ChatVertexAI(model_name="gemini-2.5-flash")
            
            print("--- Clientes de Google Cloud inicializados correctamente. ---")
        except Exception as e:
            print(f"--- !!! ERROR CRÍTICO durante la inicialización: {e} ---")
            traceback.print_exc()
            clients = {}
    return clients

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def delete_embedded_documents_by_doc_id(doc_id: str) -> bool:
    firestore_client = get_clients().get('firestore')
    if not firestore_client: return False
    try:
        FIRESTORE_COLLECTION = "pdf_embeded_documents"
        doc_id_filter = FieldFilter('metadata.doc_id', '==', doc_id)
        query = firestore_client.collection(FIRESTORE_COLLECTION).where(filter=doc_id_filter)
        docs_to_delete = list(query.stream())
        if not docs_to_delete:
            print(f"No se encontraron documentos existentes con doc_id: {doc_id}")
            return True
        batch = firestore_client.batch()
        deleted_count = 0
        for doc in docs_to_delete:
            batch.delete(doc.reference)
            deleted_count += 1
            if deleted_count % 400 == 0:
                batch.commit()
                batch = firestore_client.batch()
        if deleted_count > 0 and deleted_count % 400 != 0:
            batch.commit()
        print(f"Eliminados {len(docs_to_delete)} documentos con doc_id: {doc_id}")
        return True
    except Exception as e:
        print(f"Error al eliminar documentos: {e}")
        return False

def extract_pdf_metadata_with_llm(file_bytes: bytes) -> Dict[str, Any]:
    llm = get_clients().get('llm')
    if not llm: return {"parsed": None, "raw_output": "LLM client not initialized"}
    
    pdf_base64 = base64.b64encode(file_bytes).decode("utf-8")
    system_prompt = (
        "Eres un asistente experto en análisis de documentos sobre derechos humanos..." # (Tu prompt largo aquí)
    )
    parser = JsonOutputParser()
    try:
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Extrae los metadatos del siguiente documento en PDF."},
                {"type": "image_url", "image_url": f"data:application/pdf;base64,{pdf_base64}"}
            ]
        )
        response = llm.invoke([SystemMessage(content=system_prompt), message])
        raw_output = response.content if isinstance(response.content, str) else ""
        # ... (resto de la lógica de parseo)
    except Exception as e:
        # ... (resto del manejo de errores)
    # ... (El resto de esta función y las demás funciones auxiliares no cambian) ...

# ==============================================================================
# ENDPOINT PRINCIPAL AUTOMATIZADO
# ==============================================================================
@app.route("/", methods=["POST"])
def handle_gcs_event():
    """
    Este endpoint único recibe y procesa los eventos de Cloud Storage.
    """
    try:
        clients = get_clients()
        storage_client = clients.get('storage')
        if not storage_client:
            print("Error: Storage client no inicializado.")
            return "Error interno", 500

        # El cuerpo de la petición es un evento CloudEvent enviado por el activador
        event = request.get_json()
        
        # Extraemos los datos del evento de Cloud Storage
        bucket_name = event.get("bucket")
        file_id = event.get("name")
        
        if not bucket_name or not file_id:
            print(f"Evento inválido recibido: {event}")
            return "Evento inválido", 400

        print(f"Archivo detectado: {file_id} en bucket: {bucket_name}")

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_id)
        if not blob.exists():
            print(f"Archivo {file_id} no encontrado.")
            return f"Archivo no encontrado", 404
            
        file_bytes = blob.download_as_bytes()
        
        # Aquí iría la llamada a tu lógica de procesamiento principal
        # _process_and_embed_pdf_content(file_bytes, file_id, {})
        
        print(f"Éxito al procesar {file_id}.")
        return "Procesado con éxito", 200

    except Exception as e:
        print(f"Error inesperado procesando el evento: {traceback.format_exc()}")
        return f"Error inesperado: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
