class Config:
    num_epochs = 1000
    batch_size = 64
    rnn_size = 100
    encoder_embedding_size = 40
    decoder_embedding_size = 40
    learning_rate = 0.001
    max_grad_norm = 4.0
    encoder_vocab_size = 8941
    decoder_vocab_size = 10949
    start_token = 2
    end_token = 3
    use_beamsearch_decode = True
    max_decode_step = 20
    beam_width = 10
    optimizer = 'Adam'
    source_max = 15
    target_max = 15