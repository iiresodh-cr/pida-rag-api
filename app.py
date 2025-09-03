import os
from flask import Flask, jsonify
from rag_blueprint import firebase_rag_bp # Importamos la lógica desde el otro archivo

app = Flask(__name__)

# Registramos el blueprint. Todas las rutas definidas en él
# ahora estarán disponibles bajo el prefijo /api/rag
app.register_blueprint(firebase_rag_bp, url_prefix='/api/rag')

@app.route("/")
def index():
    """Endpoint raíz para verificar que el servicio está funcionando."""
    return jsonify(
        status="ok",
        message="PIDA RAG API is running."
    ), 200

if __name__ == "__main__":
    # Esto permite ejecutar la aplicación localmente para pruebas.
    # Cloud Run usará el comando del Dockerfile (gunicorn).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
