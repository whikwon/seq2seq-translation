PAD = 0
UNK = 1

class Config:
    num_epochs = 30
    batch_size = 128
    rnn_size = 50
    encoder_embedding_size = 15
    decoder_embedding_size = 15
    learning_rate = 0.001
    max_grad_norm = 5.0
    use_dropout = True
    dropout_rate = 0.5
    encoder_vocab_size = 30
    decoder_vocab_size = 30
    start_token = 2
    end_token = 3
    use_beamsearch_decode = False
    max_decode_step = 10
    beam_width = 20
