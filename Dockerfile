FROM pytorch/pytorch:latest

WORKDIR /app

# Copiar los archivos al contenedor
COPY modelo_regresion_aleatoria.py /app/modelo_regresion_aleatoria.py

# Copiar el conjunto de datos al contenedor
COPY sensor-data.csv /app/sensor-data.csv

# Copiar el archivo de requisitos al contenedor
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
# Instalar los requisitos
RUN pip install -r /app/requirements.txt

# Ejecutar el script de inicio cuando se inicie el contenedor
CMD ["python", "/app/modelo_regresion_aleatoria.py"]


