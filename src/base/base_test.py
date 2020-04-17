import tensorflow as tf


class BaseTest:
    def __init__(self, sess, model, train_loader, eval_loader, test_loader, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        # self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.sess.run(self.init)

    def train_step(self):
        raise NotImplementedError

    def eval_step(self):
        raise NotImplementedError

    def test_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
