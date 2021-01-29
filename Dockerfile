FROM ubuntu:18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y \
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
        tree \
        dstat \
        parallel \
        moreutils \
        unzip \
        tmux \
        build-essential \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        sox \
        ffmpeg \
        julius \
        open-jtalk \
        open-jtalk-mecab-naist-jdic \
        hts-voice-nitech-jp-atr503-m001 \
        swig \
        libsndfile1-dev \
        libasound2-dev \
        perl \
        pulseaudio \
        git-lfs \
        && \
    apt-get clean

# miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda install -y python=3.8.5

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

CMD ["bash"]
