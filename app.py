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
# (Aquí van todas tus funciones auxiliares: compute_hash, delete_embedded_documents_by_doc_id,
#  extract_pdf_metadata_with_llm, split_pdf_into_documents, etc.
#  Asegúrate de que el código de estas funciones esté aquí)

# ...

# ==============================================================================
# ENDPOINTS DE LA API
# ==============================================================================

@app.route("/")
def index():
    return jsonify(status="ok", message="PIDA RAG API is running."), 200

@app.route("/api/rag/process-pdf-from-bucket", methods=["POST"])
def process_pdf_from_bucket_endpoint():
    try:
        clients = get_clients()
        storage_client = clients.get('storage')

        if not storage_client:
            return jsonify(status="error", reason="Storage client not initialized"), 500

        if not request.is_json:
            return jsonify(status="error", reason="Content-Type must be application/json"), 400
        
        data = request.get_json()
        bucket_name = data.get("bucket_name")
        file_id = data.get("file_id")

        if not bucket_name or not file_id:
            return jsonify(status="error", reason="Missing 'bucket_name' or 'file_id'"), 400
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_id)
        
        if not blob.exists():
            return jsonify(status="error", reason=f"File '{file_id}' not found in bucket '{bucket_name}'"), 404
        
        file_bytes = blob.download_as_bytes()
        result = _process_and_embed_pdf_content(file_bytes, file_id, data.get("metadata", {}))
        
        if result.get("status") == "ok":
            return jsonify(result.get("data")), result.get("code")
        else:
            return jsonify(status="error", reason=result.get("reason")), result.get("code")

    except Exception as e:
        print(traceback.format_exc())
        return jsonify(status="error", reason=f"Error inesperado: {str(e)}"), 500

# (Aquí irían los demás endpoints)

# ...

# El bloque final para ejecución local
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
