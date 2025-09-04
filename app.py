# app.py (Versión Final con Inicialización Segura "Lazy Loading")

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

app = Flask(__name__)

# --- Declaración de Clientes Globales (sin inicializar) ---
clients = {}

def get_clients():
    """
    Inicializa los clientes de Google Cloud de forma 'perezosa' (solo una vez).
    Esta función es segura para usar en entornos de producción con Gunicorn.
    """
    global clients
    if 'firestore' not in clients: # Solo inicializa si un cliente clave falta
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
def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def delete_embedded_documents_by_doc_id(doc_id: str) -> bool:
    firestore_client = get_clients().get('firestore')
    if not firestore_client: return False
    try:
        # ... (código sin cambios)
    except Exception as e:
        # ... (código sin cambios)

def extract_pdf_metadata_with_llm(file_bytes: bytes) -> Dict[str, Any]:
    llm = get_clients().get('llm')
    if not llm: return {"parsed": None, "raw_output": "LLM client not initialized"}
    # ... (código sin cambios)

def split_pdf_into_documents(doc_id: str, file_bytes: bytes, base_metadata: Dict[str, Any]) -> List[Document]:
    # ... (código sin cambios)

def generate_chunk_ids(documents: List[Document]) -> List[str]:
    # ... (código sin cambios)

def _process_and_embed_pdf_content(file_bytes: bytes, filename: str, incoming_metadata: Dict[str, Any]) -> Dict[str, Any]:
    clients = get_clients()
    if not clients: raise Exception("Los clientes de GCP no se pudieron inicializar.")
    
    doc_id = compute_hash(file_bytes)
    delete_embedded_documents_by_doc_id(doc_id)
    # ... (resto de la función)
    vector_store = FirestoreVectorStore(
        collection="pdf_embeded_documents",
        embedding_service=clients.get('embedding'),
        client=clients.get('firestore')
    )
    # ... (resto de la función)

def format_search_results(documents: List[Document]) -> str:
    # ... (código sin cambios)

def perform_similarity_search(query: str, k: int, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Document]:
    clients = get_clients()
    if not clients: raise Exception("Los clientes de GCP no se pudieron inicializar.")
    # ... (resto de la función)

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

        # ... (resto del endpoint sin cambios, usando `storage_client` local)
        # ...
    except Exception as e:
        # ... (resto del endpoint sin cambios)

# ... (El resto de tus endpoints) ...

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
