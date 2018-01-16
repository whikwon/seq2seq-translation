import codecs
import os
import re
import numpy as np

PROCESSED_PATH = './data'
PAD = 0
UNK = 1
THRESHOLD = 2
end_token = 3


def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens."""
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def build_vocab(filename, normalize_digits=True):
    """Build vocab dictionary"""
    in_path = os.path.join(PROCESSED_PATH, filename)
    out_path = os.path.join(PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))
    vocab = {}
    f = codecs.open(in_path, 'r', encoding='utf-8')
    for line in f.readlines():
        for token in basic_tokenizer(line, normalize_digits):
            if not token in vocab:
                vocab[token] = 0
            vocab[token] += 1
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n') # <pad>: 0
        f.write('<unk>' + '\n') # <unk>: 1
        f.write('<s>' + '\n')   # <s>: 2
        f.write('<\s>' + '\n')  # <\s>: 3
        index = 4
        for word in sorted_vocab:
            if vocab[word] < THRESHOLD:
                with open('vocab.txt', 'w') as cf:
                    if filename[-3:] == 'enc':
                        cf.write('ENC_VOCAB = ' + str(index) + '\n')
                    else:
                        cf.write('\n' + 'DEC_VOCAB = ' + str(index) + '\n')
                break
            f.write(word + '\n')
            index += 1


def load_vocab(vocab_path):
    """Read vocab from text file"""
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()
    return words, {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    """Convert a sentence to ids"""
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]


def id2sentence(vocab, ids):
    """Convert ids to a sentence"""
    reversed_vocab = {j: i for i, j in vocab.items()}
    words = [reversed_vocab[i] for i in ids]
    return ' '.join(words)


def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode
    _, vocab = load_vocab(os.path.join(PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(PROCESSED_PATH, in_path), 'r')
    out_file = open(os.path.join(PROCESSED_PATH, out_path), 'w')
    lines = in_file.read().splitlines()
    for line in lines:
        ids = sentence2id(vocab, line)
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')


def load_data(encoder_filename, decoder_filename, source_max, target_max):
    """Load data from encoder & decoder text files. Returning encoder/decoder
    source and length as arrays"""
    encoder_file = open(os.path.join(PROCESSED_PATH, encoder_filename), 'r')
    decoder_file = open(os.path.join(PROCESSED_PATH, decoder_filename), 'r')
    encoder, decoder = encoder_file.readline(), decoder_file.readline()
    encoder_data = []
    decoder_data = []
    encoder_data_length = []
    decoder_data_length = []
    i = 0
    while encoder and decoder:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        encode_ids = [int(id_) for id_ in encoder.split()]
        decode_ids = [int(id_) for id_ in decoder.split()]
        if len(encode_ids) <= source_max and len(decode_ids) < target_max \
            and len(encode_ids) > 0 and len(decode_ids) > 0:
            encoder_data.append(_pad_input(encode_ids, source_max))
            decode_ids.append(end_token)
            decoder_data.append(_pad_input(decode_ids, target_max))
            encoder_data_length.append(len(encode_ids))
            decoder_data_length.append(len(decode_ids))
        encoder, decoder = encoder_file.readline(), decoder_file.readline()
        i += 1
    return np.array(encoder_data, dtype=np.int32), np.array(encoder_data_length, dtype=np.int32), \
           np.array(decoder_data, dtype=np.int32), np.array(decoder_data_length, dtype=np.int32)


def _pad_input(x, size):
    """Sequence padding"""
    return np.pad(x, (0, size - len(x)), 'constant')


def get_batch(encoder_data, encoder_data_length, decoder_data, decoder_data_length,
              batch_size, num_epochs):
    """Return batch to feed into the model."""
    num_data = len(encoder_data)
    for i in range(num_epochs):
        indices = np.random.choice(num_data, num_data, False)
        encoder_data = encoder_data[indices]
        decoder_data = decoder_data[indices]
        encoder_data_length = encoder_data_length[indices]
        decoder_data_length = decoder_data_length[indices]

        for j in range(num_data // batch_size):
            batch_encoder_inputs_length = encoder_data_length[j * batch_size: (j + 1) * batch_size]
            max_encoder_length = max(batch_encoder_inputs_length)
            batch_encoder_inputs = encoder_data[j * batch_size: (j + 1) * batch_size][:, :max_encoder_length]
            batch_decoder_inputs_length = decoder_data_length[j * batch_size: (j + 1) * batch_size]
            max_decoder_length = max(batch_decoder_inputs_length)
            batch_decoder_inputs = decoder_data[j * batch_size: (j + 1) * batch_size][:, :max_decoder_length]
            yield batch_encoder_inputs, batch_encoder_inputs_length, \
                  batch_decoder_inputs, batch_decoder_inputs_length


def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    return input()


def main():
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('test', 'enc')
    token2id('train', 'dec')
    token2id('test', 'dec')


if __name__ == '__main__':
    main()