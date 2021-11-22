import tensorflow as tf
import os
import numpy as np
from scripts import cloud_loader
from scripts import pointnet_wrapper
import sys
import pickle

tf.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def evaluate_equal(vars1, vars2):
    cond = list()
    for var1, var2 in zip(vars1, vars2):
        cond.append(np.all(session.run(var1) == session.run(var2)))
    return np.all(cond)


custom_op = ["/workspace/scripts/PointNet2/tf_ops/3d_interpolation/tf_interpolate_so.so",
            "/workspace/scripts/PointNet2/tf_ops/3d_interpolation/zero_out.so",
            "/workspace/scripts/PointNet2/tf_ops/grouping/tf_grouping_so.so",
            "/workspace/scripts/PointNet2/tf_ops/sampling/tf_sampling_so.so" ]

loaded_op = [tf.load_op_library(op) for op in custom_op]    

num_cores = 4
batch_size = 1

test_data_path = [i.tolist() for i in np.load("/workspace/scripts/test_data_path.npy", allow_pickle=True)]

test_labels = [i.tolist() for i in np.load("/workspace/scripts/test_label.npy", allow_pickle=True)]

os.chdir("/workspace/data/")
test_data_paths = [paths_and_handpose[:2] for paths_and_handpose in test_data_path]

test_cloud_indices = [paths_and_handpose[3:5] for paths_and_handpose in test_data_path]


test_handposes_float = [[float(i) for i in paths_and_handpose[2].split(' ')] for paths_and_handpose in test_data_path]

# Here!
test_path_label_ds = tf.data.Dataset.from_tensor_slices((test_data_paths[:batch_size], test_handposes_float[:batch_size], test_cloud_indices[:batch_size], test_labels[:batch_size]))


cloud_loader = cloud_loader.CloudLoader(1024, 3)

test_ds = test_path_label_ds.map(cloud_loader.load_and_preprocess_cloud_from_path_label, num_parallel_calls=num_cores)

test_ds = test_ds.batch(batch_size).prefetch(25)

test_iter = tf.data.Iterator.from_structure(test_ds.output_types, test_ds.output_shapes)
next_test_element = test_iter.get_next()
test_init_op = test_iter.make_initializer(test_ds)

learning_rate = 0.0001
decay_step = 200000
decay_rate = 0.7
bn_init_decay = 0.5
bn_decay_decay_step = float(decay_step)
bn_decay_decay_rate = 0.5
bn_decay_clip = 0.99

# Network

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


session =  tf.Session(config=config)
init = tf.global_variables_initializer()
session.run(init)

session.run(test_init_op)

net = pointnet_wrapper.PointNet(batch_size, 1024, 3, learning_rate, decay_step, decay_rate, \
bn_init_decay, bn_decay_decay_step, bn_decay_decay_rate, bn_decay_clip, num_labels=5)

net_vars = {v.name:v for v in tf.global_variables()}

with open("/workspace/output/2021_11_04_14_50_53/save_weights.pkl", "rb") as f:
    data = pickle.load(f)

for key, value in data.items():
    if key in net_vars:
        session.run(net_vars[key].assign(value))
    else:
        raise ValueError("{} not in net_vars".format(key))



data, labels = session.run(next_test_element)

"""
Note:
graph = tf.get_default_graph()
ops = [op for op in graph.get_operations()]

Extract ops, tensors, placeholders:
https://stackoverflow.com/questions/36883949/in-tensorflow-get-the-names-of-all-the-tensors-in-a-graph
"""

feed_dict = {net.ops['pointclouds_pl']: data,
net.ops['labels_pl']: labels,
net.ops['is_training_pl']: False}

print labels
print data




step, loss_val, pred_val, softmax_out = session.run([
net.ops['step'],
net.ops['loss'],
net.ops['pred'],
net.ops['softmax']], feed_dict=feed_dict)

print softmax_out