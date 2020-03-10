import tensorflow as tf
import os


def build_disc(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # Layer 1: Embedding layer to transform indices into dense vectors
        #   of a fixed embedding size
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),

        # Layer 2: LSTM with `rnn_units` number of units.
        # TODO: Call the LSTM function defined above to add this layer.
        tf.keras.layers.LSTM(
            units=rnn_units,
            return_sequences=False,
            recurrent_initializer='glorot_uniform',
            recurrent_activation='sigmoid',
            stateful=True,  # TODO this was true. why?
            dropout=0.4
        ),

        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
        #   into the vocabulary size.
        # TODO: Add the Dense layer.

        tf.keras.layers.Dense(1)
    ])
    model.summary()
    return model

class Discriminator(object):

    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size, checkpoint_dir, learning_rate=1e-4):
        self.model = build_disc(vocab_size, embedding_dim, rnn_units, batch_size)
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "disc/my_ckpt")

    def load_weights(self):
        try:
            self.model.load_weights(self.checkpoint_prefix)
            print('loaded weights for discriminator')
        except:
            print('could not find weights to load for discriminator')

    @tf.function
    def discriminate(self, samples):
        scores = self.model(samples)
        ypred_for_auc = tf.nn.sigmoid(scores) #TODO thgis was softmax in the original
        predictions = tf.argmax(scores, 1, name="predictions")
        self.model.reset_states()
        return ypred_for_auc

    @tf.function
    def get_loss(self, real_output, fake_output, l2_reg_lambda=0.0):
        # CalculateMean cross-entropy loss
        # with tf.name_scope("loss"):
        #     losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores,
        #                                                      labels=input_y)
        #     l2_loss = 0.0
        #
        #     loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        real_loss = self.cross_entropy (tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy (tf.zeros_like(fake_output), fake_output)
        # real_loss = tf.nn.softmax_cross_entropy_with_logits (real_output, tf.ones_like(real_output))
        # fake_loss = tf.nn.softmax_cross_entropy_with_logits (fake_output, tf.zeros_like(fake_output))

        total_loss = real_loss + fake_loss
        return total_loss
        # return loss

    @tf.function
    def test_step(self, fake_samples, real_samples):
        fake_scores = self.model(fake_samples)
        real_scores = self.model(real_samples)

        loss = self.get_loss(real_scores, fake_scores)
        return loss

    @tf.function
    def train_step(self, fake_samples, real_samples):
        with tf.GradientTape() as disc_tape:
            fake_scores = self.model(fake_samples)
            real_scores = self.model(real_samples)

            loss = self.get_loss(real_scores, fake_scores)

        g_grad = disc_tape.gradient(loss, self.model.trainable_variables)
        g_grad_clip, _ = tf.clip_by_global_norm(g_grad, 5.0)
        g_updates = self.optimizer.apply_gradients(
            zip(g_grad_clip, self.model.trainable_variables))
        return loss