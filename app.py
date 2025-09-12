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

app = Flask(__name__)
clients = {}

def get_clients():
    global clients
    if 'firestore' not in clients:
        print("--- Inicializando clientes de Google Cloud por primera vez... ---")
        try:
            PROJECT_ID = os.environ.get("PROJECT_ID")
            VERTEX_AI_LOCATION = os.environ.get("VERTEX_AI_LOCATION")
            MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            
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

def _process_and_embed_pdf_content(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    try:
        # ... (Esta función ya es correcta, no necesita cambios) ...
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

        reader = PdfReader(io.BytesIO(file_bytes))
        text_content = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        if not text_content:
            return {"status": "error", "reason": "No se pudo extraer texto del PDF."}

        pdf_meta = reader.metadata
        book_title = pdf_meta.title if pdf_meta.title else os.path.splitext(filename)[0].replace("_", " ").title()
        book_author = pdf_meta.author if pdf_meta.author else "Autor Desconocido"
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = text_splitter.split_text(text_content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={ "source": filename, "chunk_index": i, "title": book_title, "author": book_author }
            )
            documents.append(doc)
        
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME, embedding_service=embedding_model, client=firestore_client
        )
        
        batch_size = 100 
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)
        
        return {"status": "ok", "message": f"Archivo {filename} procesado."}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "reason": str(e)}

@app.route("/", methods=["POST"])
def handle_gcs_event():
    try:
        # ... (Esta función ya es correcta, no necesita cambios) ...
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

@app.route("/query", methods=["POST"])
def query_rag_handler():
    try:
        request_data = request.get_json()
        user_query = request_data["query"]
        clients = get_clients()
        firestore_client = clients.get('firestore')
        embedding_model = clients.get('embedding')
        COLLECTION_NAME = "pdf_embeded_documents"
        
        vector_store = FirestoreVectorStore(
            collection=COLLECTION_NAME, embedding_service=embedding_model, client=firestore_client
        )
        found_docs = vector_store.similarity_search(query=user_query, k=4)

        results = []
        for doc in found_docs:
            # --- LA CORRECCIÓN FINAL ---
            # Accedemos al diccionario de metadatos anidado
            user_metadata = doc.metadata.get("metadata", {})
            
            result_item = {
                "source": user_metadata.get("source"),
                "content": doc.page_content,
                "title": user_metadata.get("title"),
                "author": user_metadata.get("author")
            }
            results.append(result_item)
        
        return jsonify({"results": results}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error inesperado: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
