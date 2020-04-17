"""
Function: attention rnn for sentence classification
Author: Xiang.Yang
"""
from base.base_model import BaseModel
import tensorflow as tf
from models.attention import self_attention

class TSANN(BaseModel):
    def __init__(self, config):
        # can change to super().__init__() in python3, to enable multiple inheritance
        super(TSANN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.dropout_keep_prob = tf.where(self.is_training, self.config.dropout_keep_prob, 1.0)
        self.x = tf.placeholder(tf.int32, shape=[None, self.config.max_patient_len, self.config.max_visit_len], name='input_x')
        self.pos = tf.placeholder(tf.int32, shape=[None, self.config.max_patient_len], name='input_pos')
        self.x_len = tf.placeholder(tf.int32, shape=[None], name='input_x_len')
        self.y = tf.placeholder(tf.int32, shape=[None], name='input_y')

        # Get word embeddings for each token in the sentence
        word_embeddings = tf.get_variable(name="word_embeddings", dtype=tf.float32,
                                          shape=[self.config.word_vocab_size, self.config.word_embedding_dim],
                                          trainable=True)
        position_embeddings = tf.get_variable(name="position_embeddings", dtype=tf.float32,
                                              shape=[self.config.pos_vocab_size, self.config.pos_embedding_dim],
                                              trainable=True)

        visits = tf.nn.embedding_lookup(word_embeddings, self.x)
        days = tf.nn.embedding_lookup(position_embeddings, self.pos)
        visits = tf.reshape(visits, [-1, self.config.max_visit_len, self.config.word_embedding_dim])

        """visit level"""
        # attention the visit, -> [B*max_patient_len,hidden_dim]
        with tf.name_scope("code_attention"):
            visits_attention, codes_alphas = self_attention(visits, self.config.hidden_dim, time_major=False, return_alphas=True)
            visits_attention = tf.reshape(visits_attention, [-1, self.config.max_patient_len, self.config.hidden_dim])
            codes_alphas = tf.reshape(codes_alphas, [-1, self.config.max_patient_len, self.config.max_visit_len])
            
            visits_attention = tf.concat([visits_attention, days], axis=2)

        rnn_cell = self._get_cell(self.config.hidden_dim, self.config.model_version)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)

        """patient level"""
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, visits_attention, dtype=tf.float32, sequence_length=self.x_len)
        output = self.last_relevant(outputs, self.x_len)

        '''
        # if want to include the second attention layer
        with tf.name_scope("visit_attention"):
            attention_output, visits_betas = self_attention(outputs, self.config.hidden_dim, time_major=False, return_alphas=True)
            output = attention_output
        '''
        # Compute logits from the output (-1) of the LSTM
        # print(attention_output.shape)
        #
        output = tf.contrib.layers.batch_norm(output, center=True, is_training=self.is_training, scope='bn')
        logits = tf.layers.dense(output, self.config.num_classes, name='dense1')
        self.logits = logits
        # if want to output weights to get the attention
        #self.visits_weights = visits_betas
        #self.codes_weights = codes_alphas

        
        with tf.name_scope("loss"):
            l2_lambda = 0.0001
            # sparse or no sparse with one hot target
            # class_weights = tf.constant([0.1,0.9])
            #weights = tf.gather(class_weights, self.y)
            # using weighted cross entropy
            self.cross_entropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits
                                                (targets=tf.one_hot(self.y, 2), logits=logits, pos_weight=5)) 
            #self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits)) 
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            self.cross_entropy += l2_losses
            # rmsprop optimizer
            #self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate, decay=0.95). \
            #    minimize(self.cross_entropy, global_step=self.global_step_tensor)
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate). \
                minimize(self.cross_entropy, global_step=self.global_step_tensor)
         
        self.preds = tf.cast(tf.argmax(logits, 1), tf.int32)
        self.probs = tf.nn.softmax(logits)
        correct_prediction = tf.equal(self.preds, self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    @staticmethod
    def _get_cell(hidden_size, cell_type):
        if cell_type == "rnn":
            return tf.nn.rnn_cell.RNNCell(hidden_size, activation=tf.nn.leaky_relu)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.LSTMCell(hidden_size, activation=tf.nn.leaky_relu)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size, activation=tf.nn.leaky_relu)
        else:
            print("Unknown model version: {}".format(cell_type))
            return None

    @staticmethod
    def last_relevant(seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)

