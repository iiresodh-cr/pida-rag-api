# app.py (Versión Definitiva y Unificada)

import os
import hashlib
import io
import traceback
import base64
import json
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from pypdf import PdfReader
from typing import List, Dict, Any, Optional

# LangChain y Google
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# --- Creación de la Aplicación Flask ---
app = Flask(__name__)

# --- Configuración e Inicialización de Clientes ---
try:
    PROJECT_ID = os.environ.get("PROJECT_ID")
    VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION")
    LANGCHAIN_GOOGLE_GEMINI_API_KEY = os.environ.get("LANGCHAIN_GOOGLE_GEMINI_API_KEY")
    FIRESTORE_COLLECTION = "pdf_embeded_documents"

    firestore_client = firestore.Client()
    storage_client = storage.Client()
    embedding_service = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=LANGCHAIN_GOOGLE_GEMINI_API_KEY
    )
    llm = ChatVertexAI(model_name="gemini-1.5-flash-001")
    print("--- Clientes de Google Cloud inicializados correctamente. ---")
except Exception as e:
    print(f"--- !!! ERROR CRÍTICO durante la inicialización de clientes: {e} ---")

# ==============================================================================
# FUNCIONES AUXILIARES
# (Estas son las mismas funciones de antes)
# ==============================================================================

def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def delete_embedded_documents_by_doc_id(doc_id: str) -> bool:
    if not firestore_client: return False
    try:
        doc_id_filter = FieldFilter('metadata.doc_id', '==', doc_id)
        query = firestore_client.collection(FIRESTORE_COLLECTION).where(filter=doc_id_filter)
        docs_to_delete = list(query.stream())
        for doc in docs_to_delete:
            doc.reference.delete()
        print(f"Eliminados {len(docs_to_delete)} documentos con doc_id: {doc_id}")
        return True
    except Exception as e:
        print(f"Error al eliminar documentos: {e}")
        return False

# ... (Aquí irían las otras funciones auxiliares como extract_pdf_metadata_with_llm,
#      split_pdf_into_documents, etc. Debes copiarlas de tu archivo `rag_blueprint.py`
#      y pegarlas aquí sin cambios) ...

# Pega aquí el resto de las funciones auxiliares de tu `rag_blueprint.py`

# ==============================================================================
# ENDPOINTS DE LA API
# ==============================================================================

@app.route("/")
def index():
    """Endpoint raíz para verificar que el servicio está funcionando."""
    return jsonify(status="ok", message="PIDA RAG API is running."), 200

@app.route("/api/rag/process-pdf-from-bucket", methods=["POST"])
def process_pdf_from_bucket_endpoint():
    # Pega aquí el código completo de tu endpoint "process_pdf_from_bucket_endpoint"
    # que tenías en `rag_blueprint.py`, sin la línea del decorador de blueprint.
    pass # Reemplaza este 'pass' con tu código

@app.route("/api/rag/query", methods=["POST"])
def query_endpoint():
    # Pega aquí el código completo de tu endpoint "query_endpoint"
    # que tenías en `rag_blueprint.py`.
    pass # Reemplaza este 'pass' con tu código

@app.route("/api/rag/list-bucket-files", methods=["GET"])
def list_bucket_files_endpoint():
    # Pega aquí el código completo de tu endpoint "list_bucket_files_endpoint"
    # que tenías en `rag_blueprint.py`.
    pass # Reemplaza este 'pass' con tu código

@app.route("/api/rag/embed-pdf", methods=["POST"])
def embed_pdf_endpoint():
    # Pega aquí el código completo de tu endpoint "embed_pdf_endpoint"
    # que tenías en `rag_blueprint.py`.
    pass # Reemplaza este 'pass' con tu código


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
