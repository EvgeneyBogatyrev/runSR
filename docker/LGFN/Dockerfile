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
RUN apt-get install vim -y
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && /bin/bash ~/anaconda.sh -b -p /opt/conda && rm ~/anaconda.sh && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && find /opt/conda/ -follow -type f -name '*.a' -delete && find /opt/conda/ -follow -type f -name '*.js.map' -delete && /opt/conda/bin/conda clean -afy
ENV PATH=/opt/conda/bin:$PATH
RUN conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch
RUN pip install tensorboard==1.11.0 tensorboardX==1.4
RUN apt-get clean

RUN mkdir -p /opt/conda/lib/python3.7/site-packages/
RUN chmod -R 0777 /opt/conda/lib/python3.7/site-packages/

RUN mkdir -p build/bdist.linux-x86_64/egg
RUN chmod -R 0777 build

CMD ["python3", "/model/run.py"]
