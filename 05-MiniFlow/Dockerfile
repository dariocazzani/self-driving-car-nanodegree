FROM ubuntu:16.04

MAINTAINER Dario Cazzani <dariocazzani@gmail.com>

#3.4.3
ENV PYTHON_VERSION 3.5
ENV NUM_CORES 8

# Install OpenCV 3.0
# Install "add-apt-repository
RUN apt-get update ; apt-get -y install software-properties-common dpkg
# enable universe and multiverse
RUN add-apt-repository "deb http://de.archive.ubuntu.com/ubuntu/ vivid universe multiverse" ; add-apt-repository "deb http://de.archive.ubuntu.com/ubuntu/ vivid-updates universe multiverse"
RUN apt-get -y update
RUN apt-get -y install python$PYTHON_VERSION-dev wget unzip \
                      build-essential cmake git \
                      build-essential pkg-config unzip ffmpeg qtbase5-dev python-dev python3-dev python-numpy python3-numpy python-tk\
                      libopencv-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev \
                      libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev \
                      libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev \
                      libvorbis-dev libxvidcore-dev v4l-utils \
                      build-essential cmake git pkg-config libatlas-base-dev gfortran \
                      libjasper-dev libgtk2.0-dev libavcodec-dev libavformat-dev \
                      libswscale-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libv4l-dev
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install numpy matplotlib

RUN wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O opencv3.zip && \
    unzip -q opencv3.zip && mv /opencv-3.1.0 /opencv
RUN wget https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip -O opencv_contrib3.zip && \
    unzip -q opencv_contrib3.zip && mv /opencv_contrib-3.1.0 /opencv_contrib
RUN mkdir /opencv/build
WORKDIR /opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..
RUN make -j$NUM_CORES
RUN make install
RUN /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
RUN ldconfig
RUN apt-get update


RUN pip install Pillow
RUN pip install pandas
RUN pip install ipython
RUN pip install sklearn
RUN pip install scipy
WORKDIR /root

CMD ["bash"]
