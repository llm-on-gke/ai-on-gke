FROM python:3.10

RUN pip3 install 'kubeflow-training[huggingface]'
WORKDIR /app

COPY fine-tune.py .

