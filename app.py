import os
import io
import traceback
from flask import Flask, request, jsonify
from pypdf import PdfReader
from typing import Dict, Any

# LangChain y Google
import vertexai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter

# --- Creación de la Aplicación Flask ---
app = Flask(__name__)

# --- Declaración de Clientes Globales ---
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
            MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
            print(f"--- Usando el modelo Gemini: {MODEL_NAME} ---")
            
            vertexai.init(project=PROJECT_ID, location=VERTEX_AI_LOCATION)

            clients['firestore'] = firestore.Client()
            clients['storage'] = storage.Client()
            clients['embedding'] = VertexAIEmbeddings(model_name="text-embedding-004")
            clients['llm'] = ChatVertexAI(model_name=MODEL_NAME) 
            
            print("--- Clientes de Google Cloud inicializados correctamente. ---")
        except Exception as e:
            print(f"--- !!! ERROR CRÍTICO durante la inicialización de clientes: {e} ---")
            traceback.print_exc()
            clients = {}
    return clients

# ==============================================================================
# FUNCIÓN DE PROCESAMIENTO DE PDF
# ==============================================================================
def _process_and_embed_pdf_content(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Procesa el contenido de un PDF, lo divide, crea embeddings y lo guarda en Firestore.
    """
    try:
        print(f"Iniciando procesamiento para el archivo: {filename}")
        clients = get_clients()
        firestore_client = clients.get('firestore')
        embedding_model = clients.get('embedding')

        if not firestore_client or not embedding_model:
            raise Exception("Los clientes de Firestore o Embedding no se inicializaron correctamente.")
        
        COLLECTION_NAME = "pdf_embeded_documents"
        docs_ref = firestore_client.collection(COLLECTION_NAME)
        existing_docs_query = docs_ref.where(filter=FieldFilter("metadata.source", "==", filename)).limit(1).stream()
        
        if len(list(existing_docs_query)) > 0:
            print(f"El archivo '{filename}' ya ha sido procesado anteriormente. Saltando...")
            return {"status": "skipped", "message": f"El archivo {filename} ya existe en la base de datos."}

        print(f"Paso 1/4: Leyendo contenido y metadatos del PDF '{filename}'...")
        reader = PdfReader(io.BytesIO(file_bytes))
        text_content = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not text_content:
            return {"status": "error", "reason": "No se pudo extraer texto del PDF."}

        pdf_meta = reader.metadata
        book_title = pdf_meta.title if pdf_meta.title else os.path.splitext(filename)[0].replace("_", " ").title()
        book_author = pdf_meta.author if pdf_meta.author else "Autor Desconocido"
        
        print(f"Metadatos extraídos -> Título: '{book_title}', Autor: '{book_author}'")

        print("Paso 2/4: Dividiendo el texto en fragmentos...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = text_splitter.split_text(text_content)
        
        print(f"Paso 3/4: Creando {len(chunks)} objetos Document...")
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,  # Guardar solo el texto limpio
                metadata={
                    "source": filename, 
                    "chunk_index": i, 
                    "title": book_title,
                    "author": book_author
                }
            )
            documents.append(doc)
        
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME,
            embedding_service=embedding_model,
            client=firestore_client,
        )

        batch_size = 100 
        print(f"Paso 4/4: Guardando {len(documents)} documentos en Firestore en lotes de {batch_size}...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"  -> Guardando lote {int(i/batch_size) + 1} de {int(len(documents)/batch_size) + 1}...")
            vector_store.add_documents(batch)
        
        print(f"Procesamiento completado con éxito para: {filename}")
        return {"status": "ok", "message": f"Archivo {filename} procesado y añadido a la colección '{COLLECTION_NAME}'."}

    except Exception as e:
        print(f"!!! ERROR dentro de _process_and_embed_pdf_content para {filename}: {e}")
        traceback.print_exc()
        return {"status": "error", "reason": str(e)}

# ==============================================================================
# ENDPOINT DE TRIGGER DE GCS
# ==============================================================================
@app.route("/", methods=["POST"])
def handle_gcs_event():
    try:
        clients = get_clients()
        storage_client = clients.get('storage')
        if not storage_client:
            return "Error interno del servidor", 500
        event = request.get_json(silent=True)
        if not event:
            return "Petición ignorada", 204
        bucket_name = event.get("bucket")
        file_id = event.get("name")
        if not bucket_name or not file_id:
            return "Evento no válido", 204
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_id)
        if not blob.exists():
            return "Archivo no encontrado para procesar", 204
        file_bytes = blob.download_as_bytes()
        result = _process_and_embed_pdf_content(file_bytes, file_id)
        if result.get("status") == "ok":
            return "Procesado con éxito", 200
        else:
            reason = result.get("reason", "Razón desconocida")
            return f"Fallo en el procesamiento: {reason}", 200
    except Exception as e:
        return f"Error inesperado: {str(e)}", 500

# ==============================================================================
# ENDPOINT DE CONSULTA RAG (VERSIÓN CORREGIDA FINAL)
# ==============================================================================
@app.route("/query", methods=["POST"])
def query_rag_handler():
    try:
        request_data = request.get_json()
        if not request_data or "query" not in request_data:
            return jsonify({"error": "Petición inválida. Se requiere un cuerpo JSON con la clave 'query'."}), 400
        
        user_query = request_data["query"]

        clients = get_clients()
        firestore_client = clients.get('firestore')
        embedding_model = clients.get('embedding')

        if not firestore_client or not embedding_model:
            return jsonify({"error": "Error interno del servidor al inicializar clientes."}), 500

        COLLECTION_NAME = "pdf_embeded_documents"
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME,
            embedding_service=embedding_model,
            client=firestore_client,
        )

        found_docs = vector_store.similarity_search(query=user_query, k=4)

        # --- LÓGICA CORREGIDA PARA DEVOLVER TODOS LOS METADATOS ---
        results = [
            {
                "source": doc.metadata.get("source"),
                "content": doc.page_content,
                "title": doc.metadata.get("title"),
                "author": doc.metadata.get("author")
            }
            for doc in found_docs
        ]
        
        return jsonify({"results": results}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error inesperado: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
