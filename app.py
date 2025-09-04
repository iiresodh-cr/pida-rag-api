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
        "Eres un asistente experto en análisis de documentos sobre derechos humanos. "
        "Tu tarea es analizar el contenido de un documento en PDF y extraer la siguiente información en formato JSON:\n\n"
        "- title: Título completo del documento\n"
        "- authors: Lista de autores o instituciones responsables\n"
        "- publication_date: Fecha de publicación (formato YYYY-MM-DD)\n"
        "- topic: Tema central del documento\n"
        "- document_type: Tipo de documento. Elige de: [\"informe\", \"resolución\", \"sentencia\", \"ley\", \"otro\"]\n"
        "- issuing_organization: Nombre de la organización que publica\n"
        "- region_or_country: Región o país de enfoque\n"
        "- categories: Lista de categorías relevantes\n\n"
        "Si un dato no está, usa null. Responde solo con el JSON."
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
        try:
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                json_string = json_match.group(0)
                parsed = parser.parse(json_string)
                return {"parsed": parsed, "raw_output": raw_output}
            else:
                print(f"Advertencia: El LLM no devolvió un JSON válido. Salida: {raw_output}")
                return {"parsed": None, "raw_output": raw_output}
        except (OutputParserException, json.JSONDecodeError, TypeError) as e:
            print(f"Error al parsear la salida del LLM como JSON: {e}\nSalida recibida: {raw_output}")
            return {"parsed": None, "raw_output": raw_output}
    except Exception as e:
        print(f"Error al invocar LLM: {e}")
        traceback.print_exc()
        return {"parsed": None, "raw_output": None}

def split_pdf_into_documents(doc_id: str, file_bytes: bytes, base_metadata: Dict[str, Any]) -> List[Document]:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    all_docs = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text: continue
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({"page_number": page_num + 1})
            all_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
    return all_docs

def generate_chunk_ids(documents: List[Document]) -> List[str]:
    if not documents: return []
    doc_id = documents[0].metadata.get("doc_id", "unknown_doc")
    return [f"{doc_id}_p{doc.metadata.get('page_number', 0)}_{i}" for i, doc in enumerate(documents)]

def _process_and_embed_pdf_content(file_bytes: bytes, filename: str, incoming_metadata: Dict[str, Any]) -> Dict[str, Any]:
    clients = get_clients()
    if not clients:
        raise Exception("Los clientes de GCP no se pudieron inicializar.")
    
    FIRESTORE_COLLECTION = "pdf_embeded_documents"
    doc_id = compute_hash(file_bytes)
    delete_embedded_documents_by_doc_id(doc_id)
    
    print("Extrayendo metadatos con LLM...")
    extracted_metadata_llm_result = extract_pdf_metadata_with_llm(file_bytes)
    parsed_llm_metadata = extracted_metadata_llm_result.get("parsed") or {}
    
    indexing_timestamp = datetime.now(timezone.utc).isoformat()
    base_metadata = {**incoming_metadata, **parsed_llm_metadata, "doc_id": doc_id, "indexing_timestamp": indexing_timestamp}
    
    print("Dividiendo el documento en fragmentos...")
    documents = split_pdf_into_documents(doc_id, file_bytes, base_metadata)
    if not documents:
        return {"status": "error", "reason": "No se pudo extraer texto del PDF.", "code": 400}
        
    chunk_ids = generate_chunk_ids(documents)
    
    vector_store = FirestoreVectorStore(
        collection=FIRESTORE_COLLECTION,
        embedding_service=clients.get('embedding'),
        client=clients.get('firestore')
    )

    BATCH_SIZE = 400
    print(f"Guardando {len(documents)} fragmentos en Firestore en lotes de {BATCH_SIZE}...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_ids = chunk_ids[i:i + BATCH_SIZE]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"  -> Lote de {len(batch_docs)} documentos guardado en Firestore.")
    print("✅ Todos los lotes guardados en Firestore.")
        
    return {
        "status": "ok",
        "data": {"doc_id": doc_id, "chunks_created": len(documents), "filename": filename, "metadata": parsed_llm_metadata},
        "code": 200
    }

def perform_similarity_search(query: str, k: int, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Document]:
    clients = get_clients()
    if not clients: raise Exception("Los clientes de GCP no se pudieron inicializar.")
    vector_store = FirestoreVectorStore(
        collection="pdf_embeded_documents",
        embedding_service=clients.get('embedding'),
        client=clients.get('firestore')
    )
    if metadata_filters:
        firestore_filters = []
        for key, value in metadata_filters.items():
            if isinstance(value, dict) and ('start' in value or 'end' in value):
                if 'start' in value: firestore_filters.append(FieldFilter(f'metadata.{key}', '>=', value['start']))
                if 'end' in value: firestore_filters.append(FieldFilter(f'metadata.{key}', '<=', value['end']))
            else:
                firestore_filters.append(FieldFilter(f'metadata.{key}', '==', value))
        return vector_store.similarity_search(query, k=k, filters=firestore_filters)
    else:
        return vector_store.similarity_search(query, k=k)

def format_search_results(documents: List[Document]) -> str:
    if not documents: return "No se encontraron resultados relevantes."
    formatted = "Resultados de la Búsqueda:\n\n"
    for i, doc in enumerate(documents):
        formatted += f"--- Resultado {i+1} ---\n"
        formatted += f"ID del Documento: {doc.metadata.get('doc_id')}\nPágina: {doc.metadata.get('page_number')}\n"
        formatted += f"Tipo: {doc.metadata.get('document_type', 'N/A')}\n"
        formatted += f"Tema: {doc.metadata.get('topic', 'N/A')}\n"
        formatted += f"Contenido:\n\"\"\"\n{doc.page_content}\n\"\"\"\n\n"
    return formatted

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.route("/", methods=["GET", "POST"])
def main_endpoint():
    """
    Endpoint principal. Responde a GET para health checks y a POST para eventos.
    """
    if request.method == 'GET':
        return jsonify(status="ok", message="PIDA RAG API is running."), 200
    
    if request.method == 'POST':
        try:
            # Lógica para manejar el evento de GCS
            event = request.get_json(silent=True)
            if not event:
                print("Petición POST vacía o sin JSON. Ignorando.")
                return "Petición ignorada", 204

            bucket_name = event.get("bucket")
            file_id = event.get("name")
            
            if not bucket_name or not file_id:
                print(f"Evento ignorado: no es un evento de Cloud Storage válido. Evento: {event}")
                return "Evento no válido", 204

            print(f"Archivo detectado: {file_id} en bucket: {bucket_name}")

            clients = get_clients()
            storage_client = clients.get('storage')
            if not storage_client:
                print("Error: Storage client no inicializado.")
                return "Error interno del servidor", 500

            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(file_id)
            if not blob.exists():
                print(f"Archivo {file_id} no encontrado en el bucket.")
                return f"Archivo no encontrado", 204 # Devolvemos 2xx para no reintentar
                
            file_bytes = blob.download_as_bytes()
            
            result = _process_and_embed_pdf_content(file_bytes, file_id, {})
            
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

@app.route("/query", methods=["POST"])
def query_endpoint():
    """
    Endpoint para hacer consultas a la base de datos de documentos.
    """
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        query_text = data.get("query")
        if not query_text: return jsonify(status="error", reason="Falta 'query'"), 400
        
        k_results = int(data.get("k", 3))
        metadata_filters = data.get("metadata_filters")
        
        results = perform_similarity_search(query=query_text, k=k_results, metadata_filters=metadata_filters)
        formatted_results = format_search_results(results)
        
        return jsonify(status="ok", results=formatted_results), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(status="error", reason=f"Error inesperado: {str(e)}"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
