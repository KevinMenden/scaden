FROM continuumio/miniconda3

COPY environment.yml /
RUN conda env create -f /environment.yml && conda clean -a
ENV PATH /opt/conda/envs/scaden/bin:$PATH