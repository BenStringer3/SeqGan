import tensorflow as tf
import numpy as np
from data_handler import songs, get_batch, char2idx, idx2char, vocab_size, extract_songs, get_bleu_score
from rollout import Rollout
from pre_train_disc import DiscPretrainer
from discriminator import Discriminator
from pre_train_gen import GenPretrainer
from generator import Generator


np.random.seed()
import os
import datetime

#Parameters/settings
batch_size=64
embedding_dim=256
seq_len = 200 #579
gen_seq_len=600
gen_rnn_units=1024
disc_rnn_units=1024
EPOCHS=40000
PRETRAIN_EPOCHS = 4500
learning_rate = 1e-4
start_token=char2idx['\n']
rollout_num=2
gen_pretrain = False
disc_pretrain = False
load_gen_weights = True
load_disc_weights = True
save_gen_weights = False
save_disc_weights = True

disc_steps=3


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
profile_log_dir = 'logs/' + current_time + '/profiling'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# Checkpoint location:
pretrain_checkpoint_dir = './training_checkpoints'

gen = Generator(batch_size=batch_size,
                 embedding_size=embedding_dim,
                 sequence_length=seq_len,
                 vocab_size=vocab_size,
                 rnn_units=gen_rnn_units,
                start_token=start_token,
                checkpoint_dir=pretrain_checkpoint_dir)
if load_gen_weights:
    gen.load_weights()

if gen_pretrain:
    gen_pre_trainer = GenPretrainer(generator=gen,
                             pretrain_epochs=PRETRAIN_EPOCHS,
                             songs=songs,
                             char2idx=char2idx,
                             idx2char=idx2char,
                             tb_writer=train_summary_writer,
                             learning_rate=1e-4)
    print('Start pre-training generator...')
    gen_pre_trainer.pretrain(gen_seq_len, save_gen_weights)

disc = Discriminator(vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    rnn_units=gen_rnn_units,
                    batch_size=batch_size,
                     checkpoint_dir=pretrain_checkpoint_dir)
if load_disc_weights:
    disc.load_weights()

if disc_pretrain:
    disc_pre_trainer = DiscPretrainer(discriminator=disc,
                            generator=gen,
                             pretrain_epochs=PRETRAIN_EPOCHS,
                             songs=songs,
                             char2idx=char2idx,
                             idx2char=idx2char,
                             tb_writer=train_summary_writer,
                             learning_rate=1e-4)
    print('Start pre-training discriminator...')
    disc_pre_trainer.pretrain(save_disc_weights)




rollout = Rollout(  generator=gen,
                    discriminator=disc,
                    batch_size=batch_size,
                    embedding_size=embedding_dim,
                    sequence_length=seq_len,
                    start_token=start_token,
                    rollout_num=rollout_num)


for epoch in range(EPOCHS):
    fake_samples = gen.generate()
    rewards = rollout.get_reward(samples=fake_samples)
    gen_loss = gen.train_step(fake_samples, rewards)
    real_samples, _ = get_batch(seq_len, batch_size)
    disc_loss = 0
    for i in range(disc_steps):
        disc_loss += disc.train_step(fake_samples, real_samples)/disc_steps

    with train_summary_writer.as_default():
        tf.summary.scalar('gen_train_loss', gen_loss, step=epoch)
        tf.summary.scalar('disc_train_loss', disc_loss, step=epoch)
        tf.summary.scalar('total_train_loss', disc_loss + gen_loss, step=epoch)

    if epoch % 7 == 0 or epoch == 0:
        disc.model.save_weights(disc.checkpoint_prefix)
        gen.model.save_weights(disc.checkpoint_prefix)
        samples = gen.generate(gen_seq_len)
        genned_songs = extract_songs(samples)
        bleu_score = get_bleu_score(genned_songs)
        # print(idx2char[samples[0]])
        gen.model.save_weights(gen.checkpoint_prefix)

        #test disc
        fake_samples = gen.generate()
        real_samples = get_batch(seq_len, batch_size, training=False)
        disc_loss = disc.test_step(fake_samples, real_samples)

        #test gen
        gen_loss = gen.test_step()

        #record test losses
        with train_summary_writer.as_default():
            tf.summary.scalar('disc_test_loss',
                              tf.reduce_mean(disc_loss), step=epoch)
            tf.summary.scalar('gen_test_loss',
                              tf.reduce_mean(gen_loss), step=epoch)
            tf.summary.scalar('bleu_score',
                              tf.reduce_mean(bleu_score), step=epoch + gen_pretrain*PRETRAIN_EPOCHS)

