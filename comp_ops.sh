# ***COMPILE THE CUSTOM OPS IN MultiModalGrasping/scripts/PointNet2/tf_ops***
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

cd /workspace/scripts/PointNet2/tf_ops

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