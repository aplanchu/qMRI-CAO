FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04

ENV TZ=Europe/Amsterdam
ENV GIT_URL=https://github.com/GewoonMaarten/geometric-dl-dmri.git

WORKDIR /workspace

# Set timezone.
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone

# Install Python 3.9 and PIP.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y --no-install-recommends python3.9 python3.9-distutils && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 10 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm -rf get-pip.py && \
    rm -rf /var/lib/apt/lists/*

# Install the project.
COPY . .
RUN pip install -e .

ENTRYPOINT ["python"]