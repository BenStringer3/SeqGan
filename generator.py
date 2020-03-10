import tensorflow as tf
import os
from data_handler import get_batch

class Generator(object):

    def __init__(self, batch_size, embedding_size, sequence_length,
                 vocab_size, checkpoint_dir, rnn_units=32, start_token=0, learning_rate=1e-4):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "gen/my_ckpt")
        self.start_token = tf.identity(tf.constant([start_token] * self.batch_size, dtype=tf.int32))

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        self.model = tf.keras.Sequential([
            # Layer 1: Embedding layer to transform indices into dense vectors
            #   of a fixed embedding size
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_size,
                                      batch_input_shape=[batch_size, None]),

            # Layer 2: LSTM with `rnn_units` number of units.
            tf.keras.layers.LSTM(
                units=rnn_units,
                return_sequences=False,
                recurrent_initializer='glorot_uniform',
                recurrent_activation='sigmoid',
                stateful=True,  # TODO this was true. why?
                dropout=0.4
            ),

            tf.keras.layers.Dense(self.vocab_size)
        ])
        self.model.summary()

        self.g_embeddings = next(
            layer for layer in self.model.layers if
            isinstance(layer, tf.keras.layers.Embedding)).embeddings

    def load_weights(self):
        try:
            self.model.load_weights(self.checkpoint_prefix)
            print('loaded weights for generator')
        except:
            print('could not find weights to load for generator')

    @tf.function
    def generate(self, seq_len=None):
        if seq_len is None:
            seq_len = self.sequence_length
        gen_x = tf.TensorArray(dtype=tf.int32, size=seq_len,
                                             dynamic_size=False,
                               infer_shape=True)

        def _g_recurrence(i, x_t, gen_x):
            # h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            # o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            x_t = tf.reshape(x_t, [self.batch_size, 1])
            o_t = self.model(x_t)
            log_prob = tf.math.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            # x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            x_tp1 = next_token
            # gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
            #                                                  tf.nn.softmax(o_t)), 1))  # [batch_size] , prob
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, gen_x

        gen_x = gen_x.write(0, self.start_token)

        _, _,  self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2: i < seq_len,
            body=_g_recurrence,
            loop_vars=(tf.constant(1, dtype=tf.int32),
                       # tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.start_token,
                       gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length
        self.model.reset_states()
        return self.gen_x

    @tf.function
    def gen_predictions(self, x, training=False): # x in token form [batch_size, seq_length]
        # processed_x = tf.transpose(
        #     tf.nn.embedding_lookup(self.g_embeddings, x),
        #     perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        # supervised pretraining for generator
        g_predictions = tf.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        # ta_emb_x = tf.TensorArray(
        #     dtype=tf.float32, size=self.sequence_length)
        # ta_emb_x = ta_emb_x.unstack(processed_x)


        x_transposed = tf.cast(tf.transpose(x), dtype=tf.int32)
        ta_x = tf.TensorArray(
            dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(x_transposed)


        def _pretrain_recurrence(i, x_t, g_predictions):

            # h_t = self.g_recurrent_unit(x_t, h_tm1)
            # o_t = self.g_output_unit(h_t)
            x_t = tf.reshape(x_t, [self.batch_size, 1])
            o_t = self.model(x_t, training=training)
            g_predictions = g_predictions.write(i, tf.nn.softmax(o_t))  # batch x vocab_size
            x_tp1 = ta_x.read(i)
            return i + 1, x_tp1, g_predictions

        ta_x.write(0, self.start_token)
        _, _, self.g_predictions = tf.while_loop(
            cond=lambda i, _1, _2: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(1, dtype=tf.int32),
                       # tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       ta_x.read(0),
                       #  self.start_token,
                        g_predictions))

        self.g_predictions = tf.transpose(self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
        self.model.reset_states()
        return self.g_predictions

    # @tf.function
    # def gen_predictions2(self, x):
    #     x = tf.expand_dims(x, axis=-1)
    #     predictions = self.model(x)
    #     predictions = tf.nn.softmax(predictions)
    #     self.model.reset_states()
    #     return predictions
    @tf.function
    def train_step(self, samples, rewards):

        with tf.GradientTape() as tape:
            loss = self.get_loss(samples, rewards)

        g_grad, _ = tf.clip_by_global_norm(
            tape.gradient(loss, self.model.trainable_variables), 5.0)
        g_updates = self.optimizer.apply_gradients(
            zip(g_grad, self.model.trainable_variables))

        return loss

    def test_step(self):
        x, y = get_batch(self.sequence_length, self.batch_size,
                         start_with_song=False,
                         training=False)
        y_hat = self.gen_predictions(tf.constant(x))
        gen_loss = self.get_pretrain_loss(labels=x, samples=y_hat)
        return gen_loss

    @tf.function
    def get_pretrain_loss(self, labels, samples): # labels as tokens, samples as prob distr
        # labels_oh = tf.one_hot(tf.cast(labels, tf.int32), self.vocab_size,
        #                    1.0, 0.0)
        # samples_clipped = tf.clip_by_value(samples, 1e-20, 1.0)
        # loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=labels_oh, y_pred=samples_clipped, from_logits=False ))
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, samples,
                                                               from_logits=False)
        return loss

    @tf.function
    def get_loss(self, x, rewards):
        g_predictions = self.gen_predictions(x)
        loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.cast(tf.reshape(x, [-1]), tf.int32), self.vocab_size,
                           1.0, 0.0) * tf.math.log(
                    tf.clip_by_value(
                        tf.reshape(g_predictions, [-1, self.vocab_size]),
                        1e-20, 1.0)
                ), 1) * tf.reshape(rewards, [-1])
        )
        return loss