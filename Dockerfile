# Dockerfile.thesis: image from the original MLAgentBench
FROM qhwang123/researchassistant:latest

#1. We switch to root at the beginning to have full permissions.
USER root

# 2. We install system tools FIRST
# This will be cached and will not be repeated again
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*

##########################
# BASIC CONFIG
##########################
WORKDIR /MLAgentBench
SHELL ["/bin/bash", "-lc"]

##########################
# Copy current repository in the folder
##########################
COPY . .

##########################
# Install dependencies in the autogpt environment
##########################
COPY requirements_main.txt /tmp/requirements_main.txt
RUN conda env list && \
    conda run -n autogpt python -V && \
    conda run -n autogpt pip install --no-cache-dir -r /tmp/requirements_main.txt

##########################
# Create vllm_srv environment
##########################
COPY requirements_vllm_srv.txt /tmp/requirements_vllm_srv.txt
RUN conda create -y -n vllm_srv python=3.10 && \
    conda run -n vllm_srv python -m pip install --no-cache-dir -r /tmp/requirements_vllm_srv.txt 

##########################
# Quality of life for the shell
##########################
RUN echo "conda activate autogpt" >> /home/user/.bashrc

# We return to the normal user in case someone starts without --user root.
USER user


CMD ["/bin/bash"]
