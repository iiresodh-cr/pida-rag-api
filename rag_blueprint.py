# rag_blueprint.py

import hashlib
import io
import os
import traceback
import base64
import json
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
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

# --- Configuración e Inicialización de Clientes ---
# Leemos las variables de entorno una vez al iniciar.
PROJECT_ID = os.environ.get("PROJECT_ID")
VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION")
LANGCHAIN_GOOGLE_GEMINI_API_KEY = os.environ.get("LANGCHAIN_GOOGLE_GEMINI_API_KEY")
FIRESTORE_COLLECTION = "pdf_embeded_documents"

# Inicializamos los clientes de forma global para reutilizarlos.
try:
    firestore_client = firestore.Client()
    storage_client = storage.Client()
    embedding_service = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=LANGCHAIN_GOOGLE_GEMINI_API_KEY
    )
    llm = ChatVertexAI(model_name="gemini-1.5-flash-001") # Actualizado a gemini-1.5-flash
except Exception as e:
    print(f"Error crítico durante la inicialización de clientes: {e}")
    # Considera manejar este error de forma más robusta si es necesario
    firestore_client = None
    storage_client = None
    embedding_service = None
    llm = None

firebase_rag_bp = Blueprint('firebase_rag_bp', __name__)


# --- Definición de todos los endpoints y funciones auxiliares ---
# (Pega aquí el resto de tu archivo `firebase_rag_bp.py` original,
# desde la función `compute_hash` hasta el final del archivo.
# Asegúrate de reemplazar las llamadas a `config.get_firestore_client()`
# por la variable global `firestore_client` que acabamos de crear)

# Ejemplo de cómo se vería una función modificada:
def delete_embedded_documents_by_doc_id(doc_id: str) -> bool:
    """
    Deletes all document chunks associated with a specific document ID.
    """
    db = firestore_client # Usamos el cliente global
    if not db:
        print("DB_DELETE_EMBEDDED: Cliente de Firestore no inicializado.")
        return False
    try:
        doc_id_filter = FieldFilter('metadata.doc_id', '==', doc_id)
        query = db.collection(FIRESTORE_COLLECTION).where(filter=doc_id_filter)
        docs_to_delete = list(query.stream())
        for doc in docs_to_delete:
            doc.reference.delete()
        print(f"Eliminados {len(docs_to_delete)} documentos con doc_id: {doc_id}")
        return True
    except Exception as e:
        print(f"Error al eliminar documentos: {e}")
        return False

# ... (El resto de tus funciones y endpoints irían aquí,
#      con modificaciones similares si usan `config.py`) ...

# --- NOTA: Como no tengo el resto de tu código, te dejo la función de query
#      corregida como ejemplo final, que era la más importante.

def perform_similarity_search(query: str, k: int, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Document]:
    """
    Performs a vector similarity search in the Firestore vector store.
    """
    # CORRECCIÓN: FirestoreVectorStore necesita el cliente de Firestore, no el de VertexAI.
    vector_store = FirestoreVectorStore(
        collection=FIRESTORE_COLLECTION,
        embedding_service=embedding_service,
        client=firestore_client
    )

    if metadata_filters:
        # (Tu lógica de filtros aquí es correcta)
        firestore_filters = []
        for key, value in metadata_filters.items():
            if isinstance(value, dict) and ('start' in value or 'end' in value):
                if 'start' in value:
                    firestore_filters.append(FieldFilter(f'metadata.{key}', '>=', value['start']))
                if 'end' in value:
                    firestore_filters.append(FieldFilter(f'metadata.{key}', '<=', value['end']))
            else:
                 firestore_filters.append(FieldFilter(f'metadata.{key}', '==', value))
        return vector_store.similarity_search(query, k=k, filters=firestore_filters)
    else:
        return vector_store.similarity_search(query, k=k)

# (Asegúrate de copiar aquí el resto de tus endpoints y funciones auxiliares
#  desde tu archivo original)
