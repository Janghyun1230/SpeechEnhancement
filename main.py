from model import MDPHD
from test import MDPHDTest
from configuration import config
import tensorflow as tf


tf.reset_default_graph()

if __name__ == "__main__":
    if config.train:
        print("Train session")
        network=MDPHD("mdphd")
        network.build() # make graph, define loss
        network.train() # actual training
    else:
        print("Test session")
        network=MDPHDTest("mdphd")
        # ex) which_model : '{net}_{layer}_{mask}_{loss}', model_number : 300000
        network.test(model_path="./Result/{}/CheckPoint/".format(config.which_model),
                                model_number="model.ckpt-{}".format(config.model_number))
