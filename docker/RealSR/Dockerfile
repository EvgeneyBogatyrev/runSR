FROM nvidia/cuda:11.2.2-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends
RUN apt-get install -y --no-install-recommends python3-dev python3-pip python3-wheel python3-setuptools
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir "cupy-cuda112[all]==9.0.0rc1"
RUN pip install torch torchvision opencv-python
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get clean

CMD ["python3", "/model/run.py"]
