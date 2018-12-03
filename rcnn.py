import tensorflow as tf

from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper


class RCNN:
    def __init__(self, lstm_units, max_length, classes, embed_size, embedding, L2):
        self.X = tf.placeholder(tf.int32, shape=[None, max_length], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, classes], name='Y')
        self.n = tf.placeholder(tf.int32)
        self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')
        self.L2 = tf.constant(L2)
        self.loss = None
        self.accuracy = None

        with tf.name_scope('embedding'):
            self.embedded = tf.nn.embedding_lookup(embedding, self.X)

        with tf.name_scope('bi-lstm-context'):
            fw_cell = DropoutWrapper(LSTMCell(lstm_units), output_keep_prob=self.dropout_keep)
            bw_cell = DropoutWrapper(LSTMCell(lstm_units), output_keep_prob=self.dropout_keep)
            (self.context_left, self.context_right), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                                              cell_bw=bw_cell,
                                                                                              inputs=self.embedded,
                                                                                              sequence_length=
                                                                                              tf.reduce_sum(
                                                                                                  tf.sign(
                                                                                                      tf.abs(self.X)),
                                                                                                  reduction_indices=1),
                                                                                              dtype=tf.float32)

        with tf.name_scope('pad'):
            zeros = tf.zeros(shape=(2, max_length, lstm_units), dtype=tf.float32)
            zeros_emb = tf.zeros(shape=(1, max_length, embed_size), dtype=tf.float32)
            self.context_left = tf.concat((zeros, self.context_left), axis=0)
            self.context_right = tf.concat((self.context_right, zeros), axis=0)
            self.embedded = tf.concat((zeros_emb, self.embedded, zeros_emb), axis=0)

        with tf.name_scope('word-representation'):
            self.x = tf.concat([self.context_left, self.embedded, self.context_right], name='x', axis=2)
            self.W2 = tf.get_variable(name='W2', shape=[2 * lstm_units + embed_size, lstm_units],
                                      initializer=tf.initializers.random_uniform())
            self.b2 = tf.get_variable(name='b2', shape=[lstm_units],
                                      initializer=tf.initializers.random_uniform())
            self.y2 = tf.tanh(tf.einsum('aij,jk->aik', self.x, self.W2) + self.b2)

        with tf.name_scope('text-representation'):
            self.y3 = tf.layers.max_pooling1d(self.y2, pool_size=3, strides=2)
            print(self.y2, self.y3)
            self.y3 = tf.reduce_max(self.y3, axis=1)

        with tf.name_scope('output'):
            self.W4 = tf.get_variable(name='W4', shape=[lstm_units, classes],
                                      initializer=tf.initializers.random_normal())
            self.b4 = tf.get_variable(name='b4', shape=[classes],
                                      initializer=tf.initializers.random_normal())
            self.y4 = tf.nn.softmax((tf.matmul(self.y3, self.W4) + self.b4)[1:self.n + 1][:])

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.losses.log_loss(labels=self.Y, predictions=self.y4)) + self.L2

        with tf.name_scope('accuracy'):
            self.predictions = tf.argmax(self.y4, 1, name="predictions")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
