import random
import numpy as np
import tensorflow as tf
from data_handler import extract_songs, get_batch, get_bleu_score

class GenPretrainer(object):
    def __init__(self, generator, pretrain_epochs, songs, char2idx, idx2char, tb_writer=None, learning_rate=1e-4):
        self.gen = generator
        self.pretrain_epochs = pretrain_epochs
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.songs = songs
        self.seq_len = self.gen.sequence_length
        self.batch_size = self.gen.batch_size
        self.tb_writer = tb_writer
        self.char2idx = char2idx
        self.idx2char = idx2char

    # def get_pretain_batch(self):
    #     random.shuffle(self.songs)
    #     # Join our list of song strings into a single string containing all songs
    #     songs_joined = "\n\n".join(self.songs)
    #
    #     def vectorize_string(string):
    #         return np.array([self.char2idx[char] for char in string])
    #
    #     vectorized_songs = vectorize_string(songs_joined)
    #
    #     assert isinstance(vectorized_songs,
    #                       np.ndarray), "returned result should be a numpy array"
    #     # the length of the vectorized songs string
    #     n = vectorized_songs.shape[0] - 1
    #     # randomly choose the starting indices for the examples in the training batch
    #     song_start_idcs = np.where(vectorized_songs == self.char2idx['X'])[0]
    #     song_start_idcs = np.array([idx for idx in song_start_idcs if idx + self.seq_len < n])
    #     idx = np.random.choice(len(song_start_idcs), self.batch_size)
    #     idx = song_start_idcs[idx]
    #
    #     '''TODO: construct a list of input sequences for the training batch'''
    #     input_batch = [vectorized_songs[i: i + self.seq_len] for i in idx]
    #     '''TODO: construct a list of output sequences for the training batch'''
    #     output_batch = [vectorized_songs[i + 1: i + self.seq_len + 1] for i in
    #                     idx]
    #
    #     # x_batch, y_batch provide the true inputs and targets for network training
    #     x_batch = np.reshape(input_batch, [self.batch_size, self.seq_len])
    #     y_batch = np.reshape(output_batch, [self.batch_size, self.seq_len])
    #     return x_batch, y_batch

    @tf.function
    def train_step(self, x, y):
        # Use tf.GradientTape()
        with tf.GradientTape() as tape:
            '''TODO: feed the current input into the model and generate predictions'''
            y_hat = self.gen.gen_predictions(x, training=True)

            '''TODO: compute the loss!'''
            loss = self.gen.get_pretrain_loss(labels=y, samples=y_hat)

        # Now, compute the gradients
        '''TODO: complete the function call for gradient computation. 
            Remember that we want the gradient of the loss with respect all 
            of the model parameters. 
            HINT: use `model.trainable_variables` to get a list of all model
            parameters.'''
        grads = tape.gradient(loss, self.gen.model.trainable_variables)

        # Apply the gradients to the optimizer so it can update the model accordingly
        self.optimizer.apply_gradients(zip(grads, self.gen.model.trainable_variables))
        return tf.reduce_mean(loss)

    def pretrain(self, gen_seq_len=None, save_weights=True):

        if gen_seq_len is None:
            gen_seq_len = self.seq_len

        for epoch in range(self.pretrain_epochs):
            # x, y = self.get_pretain_batch()
            x, _ = get_batch(self.seq_len, self.batch_size, start_with_song=False, training=True)
            loss = self.train_step(tf.constant(x), tf.constant(x))

            if self.tb_writer is not None:
                with self.tb_writer.as_default():
                    tf.summary.scalar('gen_pre_train_loss', loss, step=epoch)
            else:
                print(loss)

            if epoch % 17 == 0 or epoch == 0:
                samples = self.gen.generate(gen_seq_len)
                genned_songs = extract_songs(samples)
                bleu_score = get_bleu_score(genned_songs)
                print(self.idx2char[samples[0]])
                if save_weights:
                    self.gen.model.save_weights(self.gen.checkpoint_prefix)
                gen_loss = self.gen.test_step()
                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        tf.summary.scalar('gen_pre_test_loss', tf.reduce_mean(gen_loss),
                                          step=epoch)
                        tf.summary.scalar('bleu_score',
                                          tf.reduce_mean(bleu_score), step=epoch)


