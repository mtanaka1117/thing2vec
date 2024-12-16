FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt update
RUN apt install -y python3 python3-pip

RUN pip install matplotlib
RUN pip install umap-learn
RUN pip install seaborn

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

WORKDIR /thing2vec
