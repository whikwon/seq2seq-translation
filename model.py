import tensorflow as tf
from tensorflow.python.layers.core import Dense

class Seq2Seq:
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.use_dropout = config.use_dropout
        self.keep_prob = 1 - config.dropout_rate
        self.keep_prob_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[], name='keep_prob')
        self.rnn_size = config.rnn_size
        self.max_grad_norm = config.max_grad_norm
        self.encoder_embedding_size = config.encoder_embedding_size
        self.decoder_embedding_size = config.decoder_embedding_size
        self.encoder_vocab_size = config.encoder_vocab_size
        self.decoder_vocab_size = config.decoder_vocab_size
        self.start_tokens = tf.ones(
            shape=[self.batch_size, 1], dtype=tf.int32) * config.start_token
        self.end_token = config.end_token
        self.end_tokens = tf.ones(
            shape=[self.batch_size, 1], dtype=tf.int32) * config.end_token
        if mode == 'decode':
            self.use_beamsearch_decode = config.use_beamsearch_decode
            self.beam_width = config.beam_width
            self.max_decode_step = config.max_decode_step

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
            # 일단 위치 맞추기 위해서 넣어주는데 나중에 빼자.
            self.decoder_targets_train = tf.concat(
                [self.decoder_inputs, self.end_tokens], axis=1)


    def build_encoder_cell(self):
        encoder_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        encoder_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        return [encoder_cell_fw, encoder_cell_bw]

    def build_encoder(self):
        with tf.variable_scope('encoder'):
            # TODO: initializer?
            encoder_embeddings = tf.get_variable(
                name='embeddings', shape=[self.encoder_vocab_size, self.encoder_embedding_size],
                dtype=tf.float32)
            encoder_inputs_embeded = tf.nn.embedding_lookup(
                encoder_embeddings, self.encoder_inputs)

            # TODO: 추가로 통과하는 layer가 있나?
            encoder_cell_fw, encoder_cell_bw = self.build_encoder_cell()
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=encoder_cell_fw, cell_bw=encoder_cell_bw, inputs=encoder_inputs_embeded,
                sequence_length=self.encoder_inputs_length, dtype=tf.float32)
            # TODO: c state, h state 구분해서 다시 LSTM tuple로 꼭 만들어줘야 하나?
            encoder_c_state = tf.concat([encoder_state[0].c, encoder_state[1].c], -1)
            encoder_h_state = tf.concat([encoder_state[0].h, encoder_state[1].h], -1)
            self.encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_c_state, encoder_h_state)
            self.encoder_outputs = tf.concat(encoder_outputs, -1)

    def build_decoder_cell(self):
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size * 2)
        attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units=self.rnn_size * 2, memory=self.encoder_outputs,
            memory_sequence_length=self.encoder_inputs_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell=decoder_cell, attention_mechanism=attn_mechanism, attention_layer_size=self.rnn_size * 2)
        decoder_initial_state = decoder_cell.zero_state(
            batch_size=self.batch_size, dtype=tf.float32).clone(cell_state=self.encoder_state)
        return decoder_cell, decoder_initial_state

    def build_decoder(self):
        with tf.variable_scope('decoder'):
            decoder_cell, decoder_initial_state = self.build_decoder_cell()
            output_layer = Dense(self.decoder_vocab_size, dtype=tf.float32, name='output_projection')
            decoder_embeddings = tf.get_variable(
                name='embeddings', shape=[self.decoder_vocab_size, self.decoder_embedding_size],
                dtype=tf.float32)

            if self.mode == 'train':
                decoder_inputs_embeded = tf.nn.embedding_lookup(
                    decoder_embeddings, self.decoder_inputs_train)

                # TODO: training
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

                # TODO: Loss
                masks = tf.sequence_mask(
                    lengths=self.decoder_inputs_length, maxlen=max_decoder_length,
                    dtype=tf.float32, name='mask')
                self.loss = tf.contrib.seq2seq.sequence_loss(
                    logits=training_logits, targets=self.decoder_targets_train, weights=masks)

                # TODO: training_op
                params = tf.trainable_variables()
                grads = tf.gradients(self.loss, params)
                clipped_grads, _ = tf.clip_by_global_norm(grads, 5.0)
                self.train_op = tf.train.AdamOptimizer(
                    self.learning_rate).apply_gradients(zip(clipped_grads, params))

            elif self.mode == 'decode':
                start_tokens = tf.reshape(self.start_tokens, [-1, ])
                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding=decoder_embeddings, start_tokens=start_tokens,
                    end_token=self.end_token)
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=decoder_cell, helper=inference_helper, initial_state=decoder_initial_state,
                    output_layer=output_layer)
                inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder, impute_finished=True, maximum_iterations=self.max_decode_step)[0]
                self.inference_logits = tf.identity(
                    inference_decoder_output.sample_id, name='predictions')

    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length):
        input_feed = self.make_input_feed(
            encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length)
        output_feed = [self.train_op, self.loss]
        outputs = sess.run(output_feed, input_feed)
        return outputs[1]

    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        input_feed = self.make_input_feed(
            encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length)
        output_feed = [self.loss]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0]

    def make_input_feed(self, encoder_inputs, encoder_inputs_length,
                        decoder_inputs, decoder_inputs_length):
        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length
        input_feed[self.decoder_inputs.name] = decoder_inputs
        input_feed[self.decoder_inputs_length.name] = decoder_inputs_length
        return input_feed

    def build(self):
        self.add_placeholders()
        # TODO: 명칭? cell로 구분?
        self.build_encoder()
        self.build_decoder()
