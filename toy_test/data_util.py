import numpy as np
import tensorflow as tf
import os

PAD = 0
UNK = 1
GO = 2
EOS = 3
start_token = GO
end_token = EOS


def read_file(path):
    """Read source from text file"""
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8', errors='ignore') as f:
        source_sentences = f.read()
    return source_sentences


def load_data(path):
    """Read source from text file and train/validation split"""
    source_sentences = read_file(path)
    vocab = make_vocab(source_sentences)
    source_letter_ids = [[vocab.get(letter, vocab['<UNK>']) for letter in line] \
                         for line in source_sentences.split('\n')]
    num_sentences = len(source_letter_ids)
    train_val_split = int(num_sentences * 0.8)
    train_source = source_letter_ids[:train_val_split]
    train_target = [list(reversed(i)) + [3] for i in train_source]
    valid_source = source_letter_ids[train_val_split:]
    valid_target = [list(reversed(i)) + [3] for i in valid_source]
    return train_source, train_target, valid_source, valid_target


def make_vocab(data):
    """Make vocab from source"""
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = set([character for line in data.split('\n') for character in line])
    int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(set_words))}
    vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}
    return vocab_to_int


def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, num_epochs, batch_size):
    """Return batch to feed into the model."""
    for i_epoch in range(num_epochs):
        for batch_i in range(0, len(sources) // batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i:start_i + batch_size]
            targets_batch = targets[start_i:start_i + batch_size]
            pad_sources_batch = np.array(pad_sentence_batch(sources_batch, PAD))
            pad_targets_batch = np.array(pad_sentence_batch(targets_batch, PAD))

            # Need the lengths for the _lengths parameters
            pad_targets_lengths = []
            for target in pad_targets_batch:
                pad_targets_lengths.append(len(target))

            pad_source_lengths = []
            for source in pad_sources_batch:
                pad_source_lengths.append(len(source))

            yield  pad_sources_batch, np.array(pad_source_lengths), pad_targets_batch, np.array(pad_targets_lengths)


def process_decoder_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return dec_input


def _get_user_input():
    """ Get user's input, which will be transformed into encoder input later """
    print("> ", end="")
    return input()


def source2id(vocab, text):
    """Convert a source to ids"""
    sequence_length = 7
    return [vocab.get(word, vocab['<UNK>']) for word in text] \
           + [vocab['<PAD>']] * (sequence_length - len(text))


def id2source(vocab, seq):
    """Convert ids to a source"""
    reversed_vocab = {j: i for i, j in vocab.items()}
    return ''.join([reversed_vocab[i] for i in seq])