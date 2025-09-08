# Dockerfile

# Usa una imagen base de Python oficial y ligera.
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app

# Copia solo el archivo de requerimientos primero para aprovechar el caché.
COPY requirements.txt .
# Instala las dependencias.
RUN pip install --no-cache-dir -r requirements.txt

# Ahora, copia todo el código de tu aplicación.
COPY . .

# Expone el puerto 8080, el estándar para Cloud Run.
EXPOSE 8080

# --- COMANDO FINAL Y DEFINITIVO ---
# Volvemos al worker estándar. Ya no necesitamos 'gevent'.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
