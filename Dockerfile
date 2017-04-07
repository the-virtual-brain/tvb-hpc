FROM ubuntu
RUN apt update -y && apt install -y build-essential git vim curl
WORKDIR /root

RUN curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Mini*.sh -b
ENV PATH /root/miniconda3/bin:$PATH
RUN conda install nomkl numba cffi ipython scipy cython && pip install sympy flake8

ADD requirements.txt /root/
RUN for req in $(cat /root/requirements.txt); do pip install $req; done

RUN git config --global core.autocrlf true

WORKDIR /root/hpc
