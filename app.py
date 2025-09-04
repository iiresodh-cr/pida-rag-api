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

# LangChain y Google
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
            clients = {} # Resetea en caso de fallo para reintentar
    return clients

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
# (El contenido de estas funciones no cambia, solo cómo obtienen los clientes)

def _process_and_embed_pdf_content(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    # El resto de las funciones auxiliares se llaman desde aquí
    
    # ... (código completo de la función)
    pass

# ==============================================================================
# ENDPOINT PRINCIPAL AUTOMATIZADO
# ==============================================================================
@app.route("/", methods=["POST"])
def handle_gcs_event():
    try:
        clients = get_clients() # Se asegura de que los clientes existan antes de usarlos
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
        
        # Aquí se llama a la lógica principal que usará los otros clientes
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
