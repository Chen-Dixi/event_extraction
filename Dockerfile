FROM tensorflow/tensorflow:1.12.0-gpu-py3
WORKDIR /EventExtractionMRC
COPY requirements.txt /EventExtractionMRC/
ENV LANG C.UTF-8
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple