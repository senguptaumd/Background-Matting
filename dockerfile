# Let image base on ubuntu 16.04
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 
ENV TZ=Europe/Berlin 
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Install python 3.6
RUN apt-get update && apt-get install -y \
                                python3.6-dev\
                                python3-pip \
                                python3-tk \
                                git libgtk2.0-dev
# Install OpenCV requirements
RUN apt-get update && apt-get install -y \
				libopencv-dev \
				python-opencv
# Install required python libraries
ADD requirements.txt . 
RUN pip3 install --upgrade pip 
RUN pip3 install --upgrade setuptools 

RUN pip3 install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r requirements.txt
# Configure environment variables
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64 
ENV CUDA_HOME=/usr/local/cuda
