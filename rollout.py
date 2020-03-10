import tensorflow as tf
import numpy as np

class Rollout(object):
    def __init__(self, generator, discriminator, batch_size, embedding_size, sequence_length, start_token=0, rollout_num=5):
        self.batch_size = batch_size
        self.discriminator = discriminator
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.generator = generator
        self.rollout_num = rollout_num
        self.g_embeddings = tf.identity(next(layer for layer in self.generator.model.layers if isinstance(layer, tf.keras.layers.Embedding)).embeddings)

        self.start_token = tf.identity(tf.constant([start_token] * self.batch_size, dtype=tf.int32))


    @tf.function
    def autobots(self, given_num, x):
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        x = tf.cast(x, tf.int32)
        ta_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(x, perm=[1, 0]))
        # def ta_emb_x(x):
        #     ta_emb_x = tf.TensorArray(
        #         dtype=tf.float32, size=self.sequence_length)
        #     processed_x_ = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, x), perm=[1, 0, 2])
        #     ta_emb_x = ta_emb_x.unstack(processed_x_)
        #     return ta_emb_x
        # ta_emb_x_ = ta_emb_x(x)

        # When current index i < given_num, use the provided tokens as the input at each time step
        def _g_recurrence_1( i, x_t, given_num, gen_x):
            x_t = tf.reshape(x_t, [self.batch_size, 1])
            _ = self.generator.model(x_t)
            # h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            # x_tp1 = ta_emb_x_.read(i)
            x_tp1 = ta_x.read(i)
            gen_x = gen_x.write(i, x_tp1)
            return i + 1, x_tp1, given_num, gen_x


        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        def _g_recurrence_2(i, x_t, given_num, gen_x):
            x_t = tf.reshape(x_t, [self.batch_size, 1])
            o_t = self.generator.model(x_t)
            # h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            # o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.math.log(tf.nn.softmax(o_t))
            next_token = tf.cast(
                tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            # x_tp1 = tf.nn.embedding_lookup(self.g_embeddings,
            #                                next_token)  # batch x emb_dim
            x_tp1 = next_token
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, given_num, gen_x

        gen_x.write(0, self.start_token)
        i, x_t, given_num, self.gen_x = tf.while_loop(
            cond=lambda i, _1, given_num, _2: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(1, dtype=tf.int32),
                       # tf.nn.embedding_lookup(self.g_embeddings, self.start_token),
                       self.start_token,
                       # ta_x.read(0),
                        given_num, gen_x))

        _, _, _,  self.gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, given_num, self.gen_x))


        gen_x2 = self.gen_x.stack()  # seq_length x batch_size
        gen_x2 = tf.transpose(gen_x2, perm=[1, 0])  # batch_size x seq_length
        self.generator.model.reset_states()
        return gen_x2

    @tf.function
    def get_unrolled_samples(self, given_num, input_x):
        samples = self.autobots(given_num, input_x)
        ypred_for_auc = self.discriminator.discriminate(samples)
        return ypred_for_auc

    def get_reward(self, samples):
        rewards = []
        for i in range(self.rollout_num):
            for given_num in tf.range(1, self.sequence_length):
                ypred_for_auc = self.get_unrolled_samples(given_num, samples)
                # ypred = np.array([item[1] for item in ypred_for_auc])
                ypred = np.array(tf.squeeze(ypred_for_auc))
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            ypred_for_auc = self.discriminator.discriminate(samples)
            # ypred = np.array([item[1] for item in ypred_for_auc])
            ypred = np.array(tf.squeeze(ypred_for_auc))
            if i == 0:
                rewards.append(ypred)
            else:
                # completed sentence reward
                rewards[self.sequence_length - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * self.rollout_num)  # batch_size x seq_length
        print(rewards)
        return rewards
