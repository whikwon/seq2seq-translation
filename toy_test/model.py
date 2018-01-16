import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest


class Seq2Seq:
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.rnn_size = config.rnn_size
        self.max_grad_norm = config.max_grad_norm
        self.encoder_embedding_size = config.encoder_embedding_size
        self.decoder_embedding_size = config.decoder_embedding_size
        self.encoder_vocab_size = config.encoder_vocab_size
        self.decoder_vocab_size = config.decoder_vocab_size
        self.optimizer = config.optimizer
        self.start_tokens = tf.ones(
            shape=[self.batch_size, 1], dtype=tf.int32) * config.start_token
        self.end_token = config.end_token
        self.end_tokens = tf.ones(
            shape=[self.batch_size, 1], dtype=tf.int32) * config.end_token
        if mode == 'decode':
            self.use_beamsearch_decode = config.use_beamsearch_decode
            self.beam_width = config.beam_width
            self.max_decode_step = config.max_decode_step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def add_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32, shape=[None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32, shape=[None, ], name='encoder_inputs_length')

        if self.mode == 'train':
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32, shape=[None, None], name='decoder_inputs')
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32, shape=[None, ], name='decoder_inputs_length')
            self.decoder_inputs_train = tf.concat(
                [self.start_tokens, self.decoder_inputs], axis=1)
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1
            self.decoder_targets_train = tf.concat(
                [self.decoder_inputs, self.end_tokens], axis=1)

    def build_single_cell(self, rnn_size):
        return tf.nn.rnn_cell.GRUCell(rnn_size)

    def build_encoder_cell(self, rnn_size):
        encoder_cell_fw = self.build_single_cell(rnn_size)
        encoder_cell_bw = self.build_single_cell(rnn_size)
        return [encoder_cell_fw, encoder_cell_bw]

    def build_encoder(self):
        with tf.variable_scope('encoder'):
            encoder_embeddings = tf.get_variable(
                name='embeddings', shape=[self.encoder_vocab_size, self.encoder_embedding_size],
                dtype=tf.float32)
            encoder_inputs_embeded = tf.nn.embedding_lookup(
                encoder_embeddings, self.encoder_inputs)
            encoder_cell_fw, encoder_cell_bw = self.build_encoder_cell(self.rnn_size)

            # bi-directional RNN
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell_fw, cell_bw=encoder_cell_bw, inputs=encoder_inputs_embeded,
                sequence_length=self.encoder_inputs_length, dtype=tf.float32)
            self.encoder_state = tf.concat(encoder_state, -1)
            self.encoder_outputs = tf.concat(encoder_outputs, -1)

    def build_decoder_cell(self):
        batch_size = self.batch_size
        encoder_outputs = self.encoder_outputs
        encoder_state = self.encoder_state
        encoder_inputs_length = self.encoder_inputs_length

        if self.mode == 'decode' and self.use_beamsearch_decode:
            encoder_state = nest.map_structure(
                lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_width), self.encoder_state)
            encoder_outputs = tf.contrib.seq2seq.tile_batch(
                self.encoder_outputs, multiplier=self.beam_width)
            encoder_inputs_length = tf.contrib.seq2seq.tile_batch(
                self.encoder_inputs_length, multiplier=self.beam_width)
            batch_size = self.batch_size * self.beam_width
        decoder_cell= self.build_single_cell(self.rnn_size * 2)

        # attention mechanism
        attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.rnn_size * 2, memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=decoder_cell, attention_mechanism=attn_mechanism, attention_layer_size=self.rnn_size * 2)
        decoder_initial_state = decoder_cell.zero_state(
            batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
        return decoder_cell, decoder_initial_state

    def build_decoder(self):
        with tf.variable_scope('decoder'):
            decoder_cell, decoder_initial_state = self.build_decoder_cell()
            output_layer = Dense(
                self.decoder_vocab_size, dtype=tf.float32, name='output_projection')
            decoder_embeddings = tf.get_variable(
                name='embeddings', shape=[self.decoder_vocab_size, self.decoder_embedding_size],
                dtype=tf.float32)

            if self.mode == 'train':
                decoder_inputs_embeded = tf.nn.embedding_lookup(
                    decoder_embeddings, self.decoder_inputs_train)

                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=decoder_inputs_embeded, sequence_length=self.decoder_inputs_length_train,
                    name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell, helper=training_helper,
                    initial_state=decoder_initial_state, output_layer=output_layer)
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)
                training_decoder_output = tf.contrib.seq2seq.dynamic_decode(
                    decoder=training_decoder, impute_finished=True, maximum_iterations=max_decoder_length)[0]
                training_logits = tf.identity(
                    training_decoder_output.rnn_output, name='logits')

                # loss
                masks = tf.sequence_mask(
                    lengths=self.decoder_inputs_length, maxlen=max_decoder_length,
                    dtype=tf.float32, name='mask')
                self.loss = tf.contrib.seq2seq.sequence_loss(
                    logits=training_logits, targets=self.decoder_targets_train, weights=masks)
                tf.summary.scalar('loss', self.loss)

                # train_op
                params = tf.trainable_variables()
                grads = tf.gradients(self.loss, params)
                clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
                if self.optimizer == 'Adam':
                    self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                        zip(clipped_grads, params), global_step=self.global_step)
                elif self.optimizer == 'Adadelta':
                    self.train_op = tf.train.AdadeltaOptimizer(self.learning_rate, rho=0.95,
                        epsilon=1e-6).apply_gradients(zip(clipped_grads, params), global_step=self.global_step)

            elif self.mode == 'decode':
                start_tokens = tf.reshape(self.start_tokens, [-1, ])
                if not self.use_beamsearch_decode:
                    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding=decoder_embeddings, start_tokens=start_tokens,
                        end_token=self.end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=decoder_cell, helper=inference_helper, initial_state=decoder_initial_state,
                        output_layer=output_layer)
                else:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=decoder_cell,
                        embedding=decoder_embeddings,
                        start_tokens=start_tokens,
                        end_token=self.end_token,
                        initial_state=decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=output_layer)
                inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder, impute_finished=False, maximum_iterations=self.max_decode_step)[0]
                if not self.use_beamsearch_decode:
                    self.inference_logits = tf.identity(
                        inference_decoder_output.sample_id, name='predictions')
                else:
                    self.inference_logits = tf.identity(
                        inference_decoder_output.predicted_ids, name='predictions')

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        input_feed = self.make_input_feed(
            encoder_inputs, encoder_inputs_length, None, None, True)
        output_feed = [self.inference_logits]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]

    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length):
        input_feed = self.make_input_feed(
            encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length, False)
        output_feed = [self.train_op, self.loss, self.global_step, self.summary_op]
        outputs = sess.run(output_feed, input_feed)
        return outputs[1:]

    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        input_feed = self.make_input_feed(
            encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length, False)
        output_feed = [self.loss]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]

    def make_input_feed(self, encoder_inputs, encoder_inputs_length,
                        decoder_inputs, decoder_inputs_length, decode):
        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length
        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length
        return input_feed

    def build(self):
        self.add_placeholders()
        self.build_encoder()
        self.build_decoder()
        self.summary_op = tf.summary.merge_all()