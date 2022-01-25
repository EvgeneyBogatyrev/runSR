FROM nvidia/cuda:11.2.2-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends
RUN apt-get install -y --no-install-recommends python3-dev python3-pip python3-wheel python3-setuptools
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir "cupy-cuda112[all]==9.0.0rc1"
RUN apt-get update -y
RUN apt-get install wget -y
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install torch torchvision
RUN pip install basicsr facexlib gfpgan
RUN pip install numpy opencv-python Pillow tqdm timm
RUN apt-get clean -y

RUN mkdir -p /usr/lib/python3.8/site-packages
RUN chmod -R 0777 /usr/lib/python3.8/site-packages

RUN mkdir -p /usr/local/lib/python3.8/dist-packages/facexlib/weights
RUN chmod -R 0777 /usr/local/lib/python3.8/dist-packages/facexlib

RUN chmod -R 0777 /usr/bin

RUN mkdir -p /usr/local/lib/python3.8/dist-packages/
RUN chmod -R 0777 /usr/local/lib/python3.8/dist-packages

CMD ["python3", "/model/run_swinir.py"]
