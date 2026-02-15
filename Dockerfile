# Dockerfile.tesis: imagen integrada para tu tesis
# Incluye:
#  - Entorno principal "autogpt" con MLAgentBench + tus extras
#  - Entorno secundario "vllm_srv" para el backend de modelos locales

FROM qhwang123/researchassistant:latest

##########################
# Config básica
##########################
# En la imagen base, el repo suele estar en /MLAgentBench
WORKDIR /MLAgentBench

# Usamos bash -lc para que "conda" funcione bien dentro de RUN
SHELL ["/bin/bash", "-lc"]

##########################
# Copiamos tu repo actual
##########################
# Esto asegura que se use el código de tu fork (no el que trae la imagen base)
COPY . .

##########################
# Instalar deps en entorno autogpt
##########################
# requirements_main.txt es el fichero limpio que hemos definido
COPY requirements_main.txt /tmp/requirements_main.txt

# Instalamos todo en el entorno conda "autogpt" que ya viene en la imagen base
RUN conda env list && \
    conda run -n autogpt python -V && \
    conda run -n autogpt pip install --no-cache-dir -r /tmp/requirements_main.txt && \
    rm /tmp/requirements_main.txt

##########################
# Crear entorno vllm_srv
##########################
# requirements_vllm_srv.txt: tu freeze actual del backend vLLM
COPY requirements_vllm_srv.txt /tmp/requirements_vllm_srv.txt

# Creamos un nuevo entorno para los modelos locales
# (elige la versión de Python con la que ya te funcionaba; 3.10 es una apuesta segura)
RUN conda create -y -n vllm_srv python=3.10 && \
    conda run -n vllm_srv python -m pip install --no-cache-dir -r /tmp/requirements_vllm_srv.txt && \
    rm /tmp/requirements_vllm_srv.txt

##########################
# Calidad de vida para el shell
##########################
# Que al abrir una shell dentro del contenedor se active autogpt por defecto
RUN echo "conda activate autogpt" >> /home/user/.bashrc

# Volvemos al usuario normal
USER user

# Comando por defecto: shell interactiva
CMD ["/bin/bash"]
