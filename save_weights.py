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


session =  tf.Session() 
init = tf.global_variables_initializer()
session.run(init)

# weight are inside the session
saver = tf.train.import_meta_graph("/workspace/output/2021_11_04_14_50_53/tf_model_epoch_35.meta")
saver.restore(session, tf.train.latest_checkpoint("/workspace/output/2021_11_04_14_50_53/"))

vars = [v for v in tf.global_variables()]

print session.run(vars[0])

save_dict = {}
for var in vars:
    save_dict[var.name] = session.run(var)

with open("/workspace/output/2021_11_04_14_50_53/save_weights.pkl", "wb") as f:
    pickle.dump(save_dict, f)
