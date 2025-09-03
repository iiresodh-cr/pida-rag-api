# app.py (Versión de Depuración)

import os
from flask import Flask, jsonify

print("--- [1] INICIANDO app.py ---")

app = Flask(__name__)
print("--- [2] Objeto Flask 'app' creado ---")

try:
    from rag_blueprint import firebase_rag_bp
    print("--- [3] IMPORTACIÓN de 'rag_blueprint' exitosa ---")
    
    app.register_blueprint(firebase_rag_bp, url_prefix='/api/rag')
    print("--- [4] REGISTRO de blueprint '/api/rag' exitoso ---")
    
    # Imprimimos todas las rutas que la aplicación conoce
    print("--- [5] Listando todas las rutas conocidas por la aplicación: ---")
    for rule in app.url_map.iter_rules():
        print(f"    -> Ruta: {rule.rule}, Endpoint: {rule.endpoint}, Métodos: {','.join(rule.methods)}")
        
except ImportError:
    print("--- !!! [ERROR FATAL] No se pudo importar 'rag_blueprint'. ¿Estás seguro de que el archivo se llama 'rag_blueprint.py'? ---")
except Exception as e:
    print(f"--- !!! [ERROR INESPERADO] Ocurrió un error durante la configuración del blueprint: {e} ---")

@app.route("/")
def index():
    """Endpoint raíz para verificar que el servicio está funcionando."""
    return jsonify(
        status="ok",
        message="PIDA RAG API is running."
    ), 200
    
print("--- [6] Configuración de app.py completada. El servidor está listo para iniciarse. ---")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
