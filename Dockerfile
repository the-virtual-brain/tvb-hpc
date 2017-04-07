FROM ubuntu
RUN apt update -y && apt install -y build-essential git vim curl
WORKDIR /root

RUN curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Mini*.sh -b
ENV PATH /root/miniconda3/bin:$PATH


RUN conda install nomkl numba cffi ipython scipy cython

# extra_pkgs="cgen genpy islpy pymbolic loopy"

ADD env/ /root/extern
RUN for pkg in cgen genpy islpy pymbolic loopy; do cd /root/extern/$pkg && python setup.py build develop && cd /root; done


# RUN git clone --recursive https://github.com/the-virtual-brain/tvb-hpc
# RUN cd tvb-hpc && PREFIX=/root/miniconda3 bash env/update-develop-packages.sh

# RUN pip install sympy flake8
# RUN git config core.autocrlf true
# RUN apt install vim-ctrlp vim-syntastic && vim-addon-manager install ctrlp syntastic
# 
# TODO rm vim git, clean up apt cache
# TODO install islpy etc from source as in hpc provisioning
