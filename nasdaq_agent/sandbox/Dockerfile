ARG PYTHON_IMAGE=python:3.13-slim-bookworm
FROM ${PYTHON_IMAGE}
RUN pip install --no-cache-dir numpy==2.3.4 pandas==2.3.3 matplotlib==3.10.7
RUN mkdir -p /output/out && chown -R 65534:65534 /output
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
USER 65534:65534
WORKDIR /output
