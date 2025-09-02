FROM dustynv/l4t-pytorch:r35.4.1

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV TORCH_CUDA_ARCH_LIST="7.2;8.7"
ENV MAX_JOBS=4 

# Build tools + headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev pkg-config \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

# Make sure build backend is ready
RUN pip install --upgrade pip setuptools wheel pybind11

# Your deps
RUN pip install imageio imageio-ffmpeg==0.4.4 
RUN pip install click scipy 
RUN pip install flask

WORKDIR /workspace
#COPY . .

ENTRYPOINT []
CMD ["python3", "stylegan_server.py"]

