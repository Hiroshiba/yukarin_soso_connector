FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y \
        locales \
        wget \
        bzip2 \
        ca-certificates \
        curl \
        git \
        gcc \
        g++ \
        cmake \
        sudo \
        htop \
        jq \
        vim \
        tree \
        dstat \
        parallel \
        moreutils \
        rsync \
        git-lfs \
        zip \
        unzip \
        tmux \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y swig libsndfile1-dev libasound2-dev && \
    apt-get clean

# environment
RUN locale-gen ja_JP.UTF-8
ENV LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8

# miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=3.8.5

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

CMD ["bash"]
