import codecs
import os
import random
import re
import numpy as np
import sys

PROCESSED_PATH = './data'
THRESHOLD = 2
end_token = 3

def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    # 구분자로 기호들이랑 따옴표를 사용한다.
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
    in_path = os.path.join(PROCESSED_PATH, filename)
    out_path = os.path.join(PROCESSED_PATH, 'vocab.{}'.format(filename[-3:]))

    # dictionary에 문장을 token으로 다 쪼개서 입력해준다.
    vocab = {}
    f = codecs.open(in_path, 'r', encoding='utf-8')
    for line in f.readlines():
        for token in basic_tokenizer(line, normalize_digits):
            if not token in vocab:
                vocab[token] = 0
            vocab[token] += 1
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    # <pad>, <unk>, <s>, <\s> 는 일단 포함한다.
    with open(out_path, 'w') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('<\s>' + '\n')
        index = 4
        for word in sorted_vocab:
            # 단어가 몇 개 있는지 알려주기 위함. sorted_vocab으로 진행하니까 마지막에
            # 개수가 1인 단어들만 몰려있을 것이고 다다르는 순간 입력하고 break.
            if vocab[word] < THRESHOLD:
                with open('vocab.txt', 'w') as cf:
                    if filename[-3:] == 'enc':
                        cf.write('ENC_VOCAB = ' + str(index) + '\n')
                    else:
                        cf.write('DEC_VOCAB = ' + str(index) + '\n')
                break
            f.write(word + '\n')
            index += 1


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        words = f.read().splitlines()

    return words, {words[i]: i for i in range(len(words))}


def sentence2id(vocab, line):
    return [vocab.get(token, vocab['<unk>']) for token in basic_tokenizer(line)]


def token2id(data, mode):
    """ Convert all the tokens in the data into their corresponding
    index in the vocabulary. """
    vocab_path = 'vocab.' + mode
    in_path = data + '.' + mode
    out_path = data + '_ids.' + mode

    # vocab에서 dictionary만 가져온다.
    _, vocab = load_vocab(os.path.join(PROCESSED_PATH, vocab_path))
    in_file = open(os.path.join(PROCESSED_PATH, in_path), 'r')
    out_file = open(os.path.join(PROCESSED_PATH, out_path), 'w')

    # 뒤에 \n을 없애기 위해서 splitlines 쓴다.
    lines = in_file.read().splitlines()
    for line in lines:
        ids = sentence2id(vocab, line)
        out_file.write(' '.join(str(id_) for id_ in ids) + '\n')


def load_data(encoder_filename, decoder_filename, source_max, target_max):
    encoder_file = open(os.path.join(PROCESSED_PATH, encoder_filename), 'r')
    decoder_file = open(os.path.join(PROCESSED_PATH, decoder_filename), 'r')

    encoder, decoder = encoder_file.readline(), decoder_file.readline()
    encoder_data = []
    decoder_data = []
    encoder_data_length = []
    decoder_data_length = []
    decoder_target_data = []
    i = 0
    while encoder and decoder:
        if (i + 1) % 10000 == 0:
            print("Bucketing conversation number", i)
        # str -> list로 만들어준다.
        encode_ids = [int(id_) for id_ in encoder.split()]
        decode_ids = [int(id_) for id_ in decoder.split()]

        # encoder, decoder 각각의 최대 크기 정한다.
        if len(encode_ids) <= source_max and len(decode_ids) <= target_max \
            and len(encode_ids) > 0 and len(decode_ids) > 0:
            encoder_data.append(_pad_input(encode_ids, source_max))
            decode_ids.append(end_token)
            decoder_data.append(_pad_input(decode_ids, target_max))
            encoder_data_length.append(len(encode_ids))
            decoder_data_length.append(len(decode_ids))
            decoder_target_data.append(_pad_input(decode_ids, target_max + 1))
        encoder, decoder = encoder_file.readline(), decoder_file.readline()
        i += 1
    return np.array(encoder_data, dtype=np.int32), np.array(encoder_data_length, dtype=np.int32), \
           np.array(decoder_data, dtype=np.int32), np.array(decoder_data_length, dtype=np.int32), \
           np.array(decoder_target_data, dtype=np.int32)


def _pad_input(x, size):
    return np.pad(x, (0, size - len(x)), 'constant')


def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in range(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                for batch_id in range(batch_size)], dtype=np.int32))
    return batch_inputs


def get_batch(encoder_data, encoder_data_length, decoder_data, decoder_data_length, decoder_target_data,
              batch_size, num_epoches):
    """ Return one batch to feed into the model """
    # pad를 조금 바꿔서 진행하자.
    num_data = len(encoder_data)
    for i in range(num_epoches):
        # epoch마다 random하게 섞어준다.
        indices = np.random.choice(num_data, num_data, False)
        encoder_data = encoder_data[indices]
        decoder_data = decoder_data[indices]
        encoder_data_length = encoder_data_length[indices]
        decoder_data_length = decoder_data_length[indices]
        decoder_target_data = decoder_target_data[indices]
        # num_data * batch_size수 만큼 iteration 돌린다.
        for j in range(num_data // batch_size):
            batch_encoder_inputs_length = encoder_data_length[j * batch_size: (j + 1) * batch_size]
            max_encoder_length = max(batch_encoder_inputs_length)
            batch_encoder_inputs = encoder_data[j * batch_size: (j + 1) * batch_size][:, :max_encoder_length]
            batch_decoder_inputs_length = decoder_data_length[j * batch_size: (j + 1) * batch_size]
            max_decoder_length = max(batch_decoder_inputs_length)
            batch_decoder_inputs = decoder_data[j * batch_size: (j + 1) * batch_size][:, :max_decoder_length]
            batch_decoder_targets = decoder_target_data[j * batch_size: (j + 1) * batch_size][:,
                                    :max_decoder_length + 1]
            yield batch_decoder_targets, batch_encoder_inputs, batch_encoder_inputs_length, \
                  batch_decoder_inputs, batch_decoder_inputs_length, \


# decode시 데이터 입력
def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    sys.stdout.flush()
    return sys.stdin.readline()


if __name__ == '__main__':
    build_vocab('train.enc')
    build_vocab('train.dec')
    token2id('train', 'enc')
    token2id('test', 'enc')
    token2id('train', 'dec')
    token2id('test', 'dec')