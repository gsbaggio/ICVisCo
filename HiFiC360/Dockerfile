# 1. Começamos com a imagem original que tem Python 3.6 e CUDA 10.0.
# Isso garante que o tensorflow-compression==1.3 VAI instalar.
FROM tensorflow/tensorflow:1.15.5-gpu-py3

# 2. Corrigimos o erro da chave GPG que esta imagem antiga tem
RUN apt-get update || true && \
    apt-get install -y --no-install-recommends gnupg dirmngr && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC && \
    apt-get update && \
    apt-get install -y git

# 3. PASSO CRÍTICO: "Troca do Motor"
# Removemos o TensorFlow 1.15 do Google (que não suporta a RTX 3090)
# E instalamos o TensorFlow 1.15 da NVIDIA (que SUPORTA a RTX 3090)
RUN pip install --upgrade pip && \
    pip uninstall -y tensorflow tensorflow-gpu && \
    pip install nvidia-pyindex && \
    pip install nvidia-tensorflow

# 4. Agora, instalamos suas dependências.
# Elas vão funcionar porque estamos no ambiente Python 3.6.
RUN pip install \
    matplotlib \
    scipy \
    tensorflow-datasets \
    tensorflow-compression==1.3 \
    git+https://github.com/google/compare_gan@19922d3004b675c1a49c4d7515c06f6f75acdcc8

# 5. Define o diretório de trabalho
WORKDIR /app

# 6. Define a variável de ambiente para CUDNN
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# 7. Comando padrão
CMD ["bash"]