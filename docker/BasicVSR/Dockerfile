FROM nvidia/cuda:11.2.2-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y --no-install-recommends
RUN apt-get install -y --no-install-recommends python3-dev python3-pip python3-wheel python3-setuptools
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir "cupy-cuda112[all]==9.0.0rc1"
RUN pip install addict future lmdb numpy opencv-python Pillow pyyaml requests scikit-image scipy tb-nightly torch>=1.7 torchvision tqdm yapf
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get clean
RUN pip install basicsr

RUN mkdir -p /usr/local/lib/python3.8/dist-packages/setuptools/command/
RUN chmod -R 0777 /usr/local/lib/python3.8

RUN mkdir -p /usr/lib/python3.8/site-packages/
RUN chmod -R 0777 /usr/lib/python3.8/site-packages/

RUN chmod -R 0777 /usr/bin

CMD ["python3", "/model/run.py"]
