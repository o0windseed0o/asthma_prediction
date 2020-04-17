import tensorflow as tf
import sys
sys.path.append('./')
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import os

def restore(ckpt_dir, prefix):
    with tf.Session() as sess:
        # load meta
        new_saver = tf.train.import_meta_graph(ckpt_dir + prefix + '.meta')
        # restore model parameters
        latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        print_tensors_in_checkpoint_file(latest_checkpoint, all_tensors=False, all_tensor_names=True, tensor_name='')
        input()

        # checkpoint_path = os.path.join(model_dir, "model.ckpt")
        reader = pywrap_tensorflow.NewCheckpointReader(latest_checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()

        for key in var_to_shape_map:
            print("tensor_name: ", key)
            input()
            print(reader.get_tensor(key))  # Remove this is you want to print only variable name

        new_saver.restore(sess, latest_checkpoint)
        # restore variables.
        logits = tf.get_collection("logits")[0]
        visits_weights = tf.get_collection("visits_betas")[0]
        codes_weights = tf.get_collection("codes_alphas")[0]
        preds = tf.get_collection("preds")[0]
        probs = tf.get_collection("probs")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("cross_entropy")[0]

if __name__ == "__main__":
    ckpt_dir = './temp_models/attcode_attrnn/asthma_exp_dual_fold1/checkpoint/'
    prefix = '-2138'
    restore(ckpt_dir, prefix)
