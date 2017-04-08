FROM ubuntu

# need these packages to install dependencies
RUN apt update -y && apt install -y build-essential git curl

# setup conda
WORKDIR /root
RUN curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Mini*.sh -b

# install conda-installable deps
ENV PATH /root/miniconda3/bin:$PATH
RUN conda install nomkl numba cffi ipython scipy cython && pip install sympy flake8

# install other deps
ADD requirements.txt /root/
RUN for req in $(cat /root/requirements.txt); do pip install $req; done

# clean up
RUN rm -rf ~/.cache ~/Miniconda3-latest-Linux-x86_64.sh
RUN apt-get remove --purge -y build-essential git curl vim $(apt-mark showauto) && apt autoremove -y && rm -rf /var/lib/apt/lists/*

# run from root dir of repo, e.g.
# docker run --rm -it -v ./:/root/hpc python -m unittest tvb_hpc.tests
WORKDIR /root/hpc

RUN pip install pudb

ENV LANG=en_US.UTF-8
RUN locale-gen $LANG
