# app.py

import os
import hashlib
import io
import traceback
import base64
import json
import re
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from pypdf import PdfReader
from typing import List, Dict, Any, Optional

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

# --- Creación de la Aplicación Flask ---
app = Flask(__name__)

# --- Declaración de Clientes Globales (sin inicializar) ---
clients = {}

def get_clients():
    """
    Inicializa los clientes de Google Cloud de forma 'perezosa' (solo una vez).
    Esta función es segura para usar en entornos de producción con Gunicorn.
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
            clients['llm'] = ChatVertexAI(model_name="gemini-2.5-flash")
            
            print("--- Clientes de Google Cloud inicializados correctamente. ---")
        except Exception as e:
            print(f"--- !!! ERROR CRÍTICO durante la inicialización de clientes: {e} ---")
            traceback.print_exc()
            clients = {}
    return clients

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def _process_and_embed_pdf_content(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    clients = get_clients()
    if not clients:
        raise Exception("Los clientes de GCP no se pudieron inicializar.")
    
    # ... (Aquí va toda tu lógica de procesamiento, como la definimos antes)

    # Ejemplo simplificado de lo que debería ir aquí
    print(f"Procesando y generando embeddings para {filename}...")
    # 1. Calcular hash
    # 2. Extraer texto
    # 3. Generar embeddings
    # 4. Guardar en Firestore
    print(f"Finalizado el procesamiento para {filename}.")
    
    # Devuelve un diccionario con los resultados
    return {
        "status": "ok",
        "data": {"filename": filename, "chunks_created": 100}, # Ejemplo
        "code": 200
    }

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
            return "Error interno del servidor", 500

        # El cuerpo de la petición es un evento CloudEvent
        event = request.get_json()
        
        # Verificamos que sea un evento de creación de archivo
        event_type = event.get("type")
        if event_type != "google.cloud.storage.object.v1.finalized":
            print(f"Evento ignorado: {event_type}")
            return "Evento ignorado", 204

        bucket_name = event["bucket"]
        file_id = event["name"]
        
        print(f"Archivo detectado: {file_id} en bucket: {bucket_name}")

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_id)
        
        if not blob.exists():
            print(f"Archivo {file_id} no encontrado en el bucket.")
            return f"Archivo no encontrado", 404
            
        file_bytes = blob.download_as_bytes()
        
        result = _process_and_embed_pdf_content(file_bytes, file_id)
        
        if result.get("status") == "ok":
            print(f"Éxito al procesar {file_id}.")
            return "Procesado con éxito", 200
        else:
            reason = result.get("reason", "Razón desconocida")
            print(f"Fallo al procesar {file_id}: {reason}")
            return f"Fallo en el procesamiento: {reason}", 500

    except Exception as e:
        print(f"Error inesperado procesando el evento: {traceback.format_exc()}")
        return f"Error inesperado: {str(e)}", 500

if __name__ == "__main__":
    # Este bloque solo se usa para desarrollo local
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
