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

def extract_pdf_metadata_with_llm(file_bytes: bytes) -> Dict[str, Any]:
    pdf_base64 = base64.b64encode(file_bytes).decode("utf-8")
    system_prompt = (
        "Eres un asistente experto en análisis de documentos sobre derechos humanos. "
        "Tu tarea es analizar el contenido de un documento en PDF (como informes, declaraciones, leyes o resoluciones) "
        "y extraer la siguiente información en formato JSON:\n\n"
        "- title: Título completo del documento\n"
        "- authors: Lista de autores o instituciones responsables (ej. [\"Comisión Interamericana de Derechos Humanos\", \"Juan Pérez\"])\n"
        "- publication_date: Fecha de publicación (formato YYYY-MM-DD si es posible, ej. \"2023-10-26\")\n"
        "- topic: Tema central del documento (ej. \"libertad de expresión\", \"violencia institucional\", \"derechos de los pueblos indígenas\", \"igualdad de género\")\n"
        "- document_type: Tipo de documento. Selecciona de la siguiente lista: [\"informe\", \"resolución\", \"declaración\", \"ley\", \"artículo académico\", \"sentencia\", \"recomendación\", \"observación general\", \"plan de acción\", \"guía\", \"manual\", \"comunicado\", \"acta\", \"tratado\", \"convención\", \"protocolo\", \"constitución\", \"política pública\", \"directriz\", \"estudio\", \"informe anual\", \"dictamen\", \"otro\"]\n"
        "- issuing_organization: Nombre de la organización o institución que publica el documento (ej. \"Oficina del Alto Comisionado de las Naciones Unidas para los Derechos Humanos\", \"Amnistía Internacional\")\n"
        "- region_or_country: Región o país al que se refiere el contenido (ej. \"América Latina\", \"Colombia\", \"Global\")\n"
        "- categories: Lista de categorías o etiquetas relevantes. Selecciona una o más de la siguiente lista: [\"Derechos Civiles y Políticos\", \"Derechos Económicos, Sociales y Culturales\", \"Derechos Colectivos\", \"Derechos de Grupos Específicos\", \"Mecanismos de Protección\", \"Derecho Internacional de los Derechos Humanos\", \"Justicia Transicional\", \"Impunity\", \"No Discriminación\", \"Libertad de Expresión\", \"Acceso a la Información\", \"Derecho a la Salud\", \"Derecho a la Educación\", \"Derecho a la Vivienda\", \"Derecho al Agua\", \"Derecho al Trabajo\", \"Derecho a la Tierra\", \"Derecho al Medio Ambiente Sano\", \"Derecho a la Vivienda Adecuada\", \"Derechos de las Mujeres\", \"Derechos de Niños, Niñas y Adolescentes\", \"Derechos de Personas con Discapacidad\", \"Derechos de Pueblos Indígenas\", \"Derechos de Personas LGBTIQ+\", \"Derechos de Migrantes y Refugiados\", \"Tortura y Otros Tratos Crueles\", \"Desaparición Forzada\", \"Ejecuciones Extrajudiciales\", \"Detención Arbitraria\", \"Seguridad Ciudadana\", \"Derecho a la Protesta\", \"Participación Ciudadana\", \"Corrupción\", \"Derechos Humanos y Empresas\", \"Derechos Humanos y Tecnología\", \"Conflictos Armados\", \"Cambio Climático y Derechos Humanos\", \"Pobreza y Derechos Humanos\"]\n\n"
        "Si algún dato no puede ser extraído, deja el campo como null. Devuelve únicamente un objeto JSON sin explicaciones."
    )
    parser = JsonOutputParser()
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage([
                {"type": "media", "data": pdf_base64, "mime_type": "application/pdf"},
                "Extrae los metadatos del documento."
            ])
        ])
        raw_output = response.content if isinstance(response.content, str) else ""
        try:
            parsed = parser.parse(raw_output)
            return {"parsed": parsed, "raw_output": raw_output}
        except OutputParserException:
            return {"parsed": None, "raw_output": raw_output}
    except Exception:
        return {"parsed": None, "raw_output": None}

def split_pdf_into_documents(doc_id: str, file_bytes: bytes, base_metadata: Dict[str, Any]) -> List[Document]:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300, separators=["\n\n", "\n", ". ", " ", ""])
    all_docs = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text: continue
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({"doc_id": doc_id, "page_number": page_num + 1})
            all_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
    return all_docs

def generate_chunk_ids(documents: List[Document]) -> List[str]:
    if not documents: return []
    doc_id = documents[0].metadata.get("doc_id", "unknown_doc")
    return [f"{doc_id}_p{doc.metadata.get('page_number', 0)}_{i}" for i, doc in enumerate(documents)]

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

def _process_and_embed_pdf_content(file_bytes: bytes, filename: str, incoming_metadata: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = compute_hash(file_bytes)
    delete_embedded_documents_by_doc_id(doc_id)
    extracted_metadata_llm_result = extract_pdf_metadata_with_llm(file_bytes)
    parsed_llm_metadata = extracted_metadata_llm_result.get("parsed") or {}
    raw_llm_output = extracted_metadata_llm_result.get("raw_output")
    indexing_timestamp = datetime.now(timezone.utc).isoformat()
    base_metadata = {**incoming_metadata, **parsed_llm_metadata, "indexing_timestamp": indexing_timestamp}
    documents = split_pdf_into_documents(doc_id, file_bytes, base_metadata)
    if not documents:
        return {"status": "error", "reason": "No se pudo extraer texto del PDF.", "code": 400}
    chunk_ids = generate_chunk_ids(documents)
    vector_store = FirestoreVectorStore(collection=FIRESTORE_COLLECTION, embedding_service=embedding_service, client=firestore_client)
    vector_store.add_documents(documents=documents, ids=chunk_ids)
    return {
        "status": "ok",
        "data": {"doc_id": doc_id, "chunks_created": len(documents), "filename": filename, "metadata": parsed_llm_metadata, "raw_llm_output": raw_llm_output},
        "code": 200
    }

# ==============================================================================
# ENDPOINTS DE LA API
# ==============================================================================

@app.route("/api/rag/process-pdf-from-bucket", methods=["POST"])
def process_pdf_from_bucket_endpoint():
    try:
        if not request.is_json: return jsonify(status="error", reason="Content-Type must be application/json"), 400
        data = request.get_json()
        bucket_name = data.get("bucket_name")
        file_id = data.get("file_id")
        additional_metadata = data.get("metadata", {})
        if not bucket_name or not file_id:
            return jsonify(status="error", reason="Missing 'bucket_name' or 'file_id'"), 400
        if not storage_client: return jsonify(status="error", reason="Storage client not initialized"), 500
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_id)
        if not blob.exists():
            return jsonify(status="error", reason=f"File '{file_id}' not found in bucket '{bucket_name}'"), 404
        file_bytes = blob.download_as_bytes()
        result = _process_and_embed_pdf_content(file_bytes, file_id, additional_metadata)
        return jsonify(result.get("data") if result.get("status") == "ok" else result), result.get("code", 500)
    except Exception as e:
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
        return jsonify(status="error", reason=f"Error inesperado: {str(e)}"), 500

# (Puedes añadir los otros endpoints como /embed-pdf y /list-bucket-files si los necesitas)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
