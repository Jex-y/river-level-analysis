FROM mcr.microsoft.com/devcontainers/anaconda:0-3

COPY environment.yml /tmp/conda-tmp/
RUN conda env create -f /tmp/conda-tmp/environment.yml && rm -rf /tmp/conda-tmp

