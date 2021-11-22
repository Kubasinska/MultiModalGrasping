FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
SHELL ["/bin/bash", "-c"]
RUN apt-get update
RUN apt-get install -y nano git
RUN touch /root/.nanorc
RUN echo "set tabsize 4" >> ~/.nanorc
RUN echo "set tabstospaces" >> ~/.nanorc
RUN apt-get install -y python2.7
RUN apt-get install -y python-pip
RUN apt-get install -y ipython

RUN python2.7 -m pip install tensorflow-gpu==1.14
RUN python2.7 -m pip install matplotlib
RUN python2.7 -m pip install protobuf==3.12.2
RUN python2.7 -m pip install scikit-learn
RUN python2.7 -m pip install pillow
RUN apt-get install -y bash
RUN apt-get install -y g++-4.8
RUN echo $SHELL
RUN mkdir "workspace"
COPY . /workspace
WORKDIR /workspace
RUN bash comp_ops.sh
#WORKDIR /workspace/scripts/PointNet2/tf_ops/3d_interpolation
#
#RUN g++-4.8 -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64/ ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2


#cd grouping/
#/usr/local/cuda-10.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++-4.8 -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64/  ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
#cd ..
#
#cd sampling
#/usr/local/cuda-10.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#g++-4.8 -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64/ ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
#cd ..