import os
import tensorflow as tf
def save_Model(model, saveDir):
    with model.graph.as_default():
            saver = tf.train.Saver()
            new_dir = os.path.join(saveDir,"SAVE_MODEL")
            saver.save(model.sess, new_dir)

    return new_dir