from data_handler import get_batch
import tensorflow as tf


class DiscPretrainer(object):
    def __init__(self, discriminator, generator, pretrain_epochs, songs, char2idx, idx2char, tb_writer=None, learning_rate=1e-4):
        self.disc = discriminator
        self.gen = generator
        self.pretrain_epochs = pretrain_epochs
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.songs = songs
        self.seq_len = self.gen.sequence_length
        self.batch_size = self.gen.batch_size
        self.tb_writer = tb_writer
        self.char2idx = char2idx
        self.idx2char = idx2char

    def pretrain(self,save_weights=True):


        for epoch in range(self.pretrain_epochs):
            fake_samples = self.gen.generate()
            real_samples = get_batch(self.seq_len, self.batch_size)
            disc_loss = self.disc.train_step(fake_samples, real_samples)

            if self.tb_writer is not None:
                with self.tb_writer.as_default():
                    tf.summary.scalar('disc_pre_train_loss', tf.reduce_mean(disc_loss), step=epoch)
            else:
                print(disc_loss)

            if epoch % 17 == 0 or epoch == 0:
                if save_weights:
                    self.disc.model.save_weights(self.disc.checkpoint_prefix)
                fake_samples = self.gen.generate()
                real_samples = get_batch(self.seq_len, self.batch_size, training=False)
                disc_loss = self.disc.test_step(real_samples=real_samples, fake_samples=fake_samples)
                if self.tb_writer is not None:
                    with self.tb_writer.as_default():
                        tf.summary.scalar('disc_pre_test_loss',
                                          tf.reduce_mean(disc_loss), step=epoch)
