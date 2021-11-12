# docker run --gpus all -v $(pwd):/workspace -it nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# sudo docker run --gpus all -v $(pwd):/workspace -p 5000:8888 -p 5001:6006 -it nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
apt-get update
apt-get install -y nano git
touch /root/.nanorc
echo "set tabsize 4" >> ~/.nanorc
echo "set tabstospaces" >> ~/.nanorc
apt-get install -y python2.7
apt-get install -y python-pip
apt-get install -y ipython

python2.7 -m pip install tensorflow-gpu==1.14
python2.7 -m pip install matplotlib
python2.7 -m pip install protobuf==3.12.2
python2.7 -m pip install scikit-learn
python2.7 -m pip install pillow

apt-get install -y g++-4.8

# ***COMPILE THE CUSTOM OPS IN MultiModalGrasping/scripts/PointNet2/tf_ops***
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

cd scripts/PointNet2/tf_ops

cd 3d_interpolation
g++-4.8 -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64/ ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
cd ..

cd grouping/
/usr/local/cuda-10.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++-4.8 -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64/  ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
cd ..

cd sampling
/usr/local/cuda-10.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++-4.8 -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64/ ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
cd ..

