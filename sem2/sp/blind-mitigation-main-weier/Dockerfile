# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:devel-gpu
MAINTAINER Jade Maï Cock “bonjour@jadecock.be”
# Miniconda install copy-pasted from Miniconda's own Dockerfile reachable 
# at: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion screen && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

RUN apt-get update && \
    apt-get install -y sudo \
    build-essential curl \
    libcurl4-openssl-dev \
    libssl-dev wget \
    python3-pip \
    unzip \
    git && \
    pip3 install --upgrade pip

RUN apt-get install -y openssh-server


ENV SHELL=/bin/bash \
    NB_USER=cock \
    NB_UID=195749 \
    NB_GROUP=DVET-unit \
    NB_GID=195749
ENV HOME=/home/$NB_USER

RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER}
RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}

# The user gets passwordless sudo
RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers

# Project setup
ARG DEBIAN_FRONTEND=noninteractive

# File Structure
RUN mkdir blind-mitigation/ && mkdir blind-mitigation/src/ && mkdir blind-mitigation/data/ && mkdir blind-mitigation/data/beerslaw/ && mkdir blind-mitigation/data/flipped_classrooms && mkdir blind-mitigation/data/tuglet &&  mkdir blind-mitigation/notebooks && mkdir blind-mitigation/experiments && mkdir blind-mitigation/experiments/temp_checkpoints/ && mkdir blind-mitigation/experiments/temp_checkpoints/training
COPY src blind-mitigation/src
COPY data/beerslaw/ blind-mitigation/data/beerslaw/
COPY data/flipped_classrooms/ blind-mitigation/data/flipped_classrooms/
COPY data/tuglet/ blind-mitigation/data/tuglet/
COPY requirements.txt blind-mitigation/requirements.txt


WORKDIR /blind-mitigation/
RUN pip3 install -r requirements.txt
# Launch feature extraction tasks
# RUN ["chmod", "+x", "./routines/run_lstms.sh"]
#CMD ["/bin/bash", "-c", "tail -f /dev/null"] # run bash to then enter the container and run the script
EXPOSE 22
EXPOSE 8888
CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"