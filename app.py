# app.py

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

# --- Configuración e Inicialización de Clientes ---
try:
    PROJECT_ID = os.environ.get("PROJECT_ID")
    VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION")
    FIRESTORE_COLLECTION = "pdf_embeded_documents"

    # Inicializa Vertex AI para establecer el contexto de autenticación
    vertexai.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)

    firestore_client = firestore.Client()
    storage_client = storage.Client()
    
    # Usa el nombre del modelo de embedding vigente
    embedding_service = VertexAIEmbeddings(model_name="text-embedding-004")
    
    # Usa el modelo correcto según tu regla de oro
    llm = ChatVertexAI(model_name="gemini-2.5-flash")

    print("--- Clientes de Google Cloud inicializados correctamente. ---")
except Exception as e:
    print(f"--- !!! ERROR CRÍTICO durante la inicialización de clientes: {e} ---")
    traceback.print_exc()

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================

def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def delete_embedded_documents_by_doc_id(doc_id: str) -> bool:
    if not firestore_client: return False
    try:
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
        if deleted_count % 400 != 0:
            batch.commit()
        print(f"Eliminados {len(docs_to_delete)} documentos con doc_id: {doc_id}")
        return True
    except Exception as e:
        print(f"Error al eliminar documentos: {e}")
        return False

def extract_pdf_metadata_with_llm(file_bytes: bytes) -> Dict[str, Any]:
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
        # Formato correcto para enviar datos multimodales a Gemini 1.5+
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Extrae los metadatos del siguiente documento en PDF."},
                {"type": "image_url", "image_url": f"data:application/pdf;base64,{pdf_base64}"}
            ]
        )
        response = llm.invoke([SystemMessage(content=system_prompt), message])
        raw_output = response.content if isinstance(response.content, str) else ""
        try:
            parsed = parser.parse(raw_output)
            return {"parsed": parsed, "raw_output": raw_output}
        except OutputParserException:
            print(f"Error al parsear JSON del LLM: {raw_output}")
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
    doc_id = compute_hash(file_bytes)
    delete_embedded_documents_by_doc_id(doc_id)
    extracted_metadata_llm_result = extract_pdf_metadata_with_llm(file_bytes)
    parsed_llm_metadata = extracted_metadata_llm_result.get("parsed") or {}
    indexing_timestamp = datetime.now(timezone.utc).isoformat()
    base_metadata = {**incoming_metadata, **parsed_llm_metadata, "doc_id": doc_id, "indexing_timestamp": indexing_timestamp}
    documents = split_pdf_into_documents(doc_id, file_bytes, base_metadata)
    if not documents:
        return {"status": "error", "reason": "No se pudo extraer texto del PDF.", "code": 400}
    chunk_ids = generate_chunk_ids(documents)
    
    vector_store = FirestoreVectorStore(collection=FIRESTORE_COLLECTION, embedding_service=embedding_service, client=firestore_client)
    # Usar lotes para añadir documentos a Firestore
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

def format_search_results(documents: List[Document]) -> str:
    if not documents: return "No se encontraron resultados relevantes."
    formatted = "Resultados de la Búsqueda:\n\n"
    for i, doc in enumerate(documents):
        formatted += f"--- Resultado {i+1} ---\n"
        formatted += f"URL: /{doc.metadata.get('doc_id')}?page={doc.metadata.get('page_number')}\n"
        formatted += f"Tipo de Documento: {doc.metadata.get('document_type', 'N/A')}\n"
        formatted += f"Tema: {doc.metadata.get('topic', 'N/A')}\n"
        formatted += f"Contenido:\n\"\"\"\n{doc.page_content}\n\"\"\"\n\n"
    return formatted

def perform_similarity_search(query: str, k: int, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Document]:
    vector_store = FirestoreVectorStore(collection=FIRESTORE_COLLECTION, embedding_service=embedding_service, client=firestore_client)
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

# ==============================================================================
# ENDPOINTS DE LA API
# ==============================================================================

@app.route("/")
def index():
    return jsonify(status="ok", message="PIDA RAG API is running."), 200

@app.route("/api/rag/process-pdf-from-bucket", methods=["POST"])
def process_pdf_from_bucket_endpoint():
    try:
        if not request.is_json: return jsonify(status="error", reason="Content-Type must be application/json"), 400
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

@app.route("/api/rag/query", methods=["POST"])
def query_endpoint():
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()
        query_text = data.get("query")
        k_results = int(data.get("k", 3))
        metadata_filters = None
        if "metadata_filters" in data:
            if isinstance(data["metadata_filters"], str): metadata_filters = json.loads(data["metadata_filters"])
            else: metadata_filters = data["metadata_filters"]
        if not query_text: return jsonify(status="error", reason="Falta 'query'"), 400
        results = perform_similarity_search(query=query_text, k=k_results, metadata_filters=metadata_filters)
        formatted_results = format_search_results(results)
        return jsonify(status="ok", results=formatted_results), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(status="error", reason=f"Error inesperado: {str(e)}"), 500

@app.route("/api/rag/embed-pdf", methods=["POST"])
def embed_pdf_endpoint():
    try:
        if "file" not in request.files:
            return jsonify(status="error", reason="Missing file"), 400
        file = request.files["file"]
        file_bytes = file.read()
        incoming_metadata = request.form.to_dict()
        result = _process_and_embed_pdf_content(file_bytes, file.filename, incoming_metadata)
        if result.get("status") == "ok":
            return jsonify(result.get("data")), result.get("code")
        else:
            return jsonify(status="error", reason=result.get("reason")), result.get("code")
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(status="error", reason=f"Error inesperado: {str(e)}"), 500
        
@app.route("/api/rag/list-bucket-files", methods=["GET"])
def list_bucket_files_endpoint():
    try:
        bucket_name = request.args.get("bucket_name")
        if not bucket_name:
            return jsonify(status="error", reason="Missing 'bucket_name' query parameter"), 400
        if not storage_client:
            return jsonify(status="error", reason="Could not initialize Google Cloud Storage client"), 500
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs()
        file_ids = [blob.name for blob in blobs if not blob.name.endswith('/')]
        return jsonify(status="ok", bucket=bucket_name, file_ids=file_ids, count=len(file_ids)), 200
    except Exception as e:
        print(traceback.format_exc())
        return jsonify(status="error", reason=f"Error inesperado: {str(e)}"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
