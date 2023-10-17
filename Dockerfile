FROM python:3.7

RUN apt-get update
RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

ARG PORT

ADD . /app/
WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY ./Ultra-Light-Fast-Generic-Face-Detector-1MB-master/MNN/model/version-RFB /app/
COPY ./Ultra-Light-Fast-Generic-Face-Detector-1MB-master/vision /app/
COPY ./Ultra-Light-Fast-Generic-Face-Detector-1MB-master/MNN/python/ultraface_py_mnn.py /app/

EXPOSE $PORT

CMD ["uvicorn", "main:app", "--port", "$PORT", "--host", "0.0.0.0"]
