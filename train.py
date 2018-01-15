import tensorflow as tf
import os
from model import Seq2Seq
from config import Config
import helper
import data_utils

FLAGS = tf.flags.FLAGS
# model
tf.flags.DEFINE_string('model_dir', './runs', 'directory for trained model')
tf.flags.DEFINE_string('model_path', './runs/model.ckpt', 'path for saving trained parameter')
tf.flags.DEFINE_integer('display_step', 20, 'steps for displaying status of training')

# checkpoint dir
if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

# config loading
config = Config()

# data loading
file_path = './data/letters_source.txt'
source_sentences = helper.load_data(file_path)
source_int_to_letter, source_letter_to_int = data_utils.extract_character_vocab(source_sentences)
source_letter_ids = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>']) for letter in line]\
                     for line in source_sentences.split('\n')]
train_source = source_letter_ids[config.batch_size:]
train_target = [list(reversed(i)) + [3] for i in train_source]
valid_source = source_letter_ids[:config.batch_size]
valid_target = [list(reversed(i)) + [3] for i in valid_source]
(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = \
    next(data_utils.get_batches(valid_target, valid_source, config.batch_size, 0, 0))

# model loading
model = Seq2Seq(config, mode='train')
model.build()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for i_epoch in range(1, config.num_epochs + 1):
    for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
            data_utils.get_batches(train_target, train_source, config.batch_size, 0, 0)):
        train_loss = model.train(sess,
                                 sources_batch,
                                 sources_lengths,
                                 targets_batch,
                                 targets_lengths)

        if batch_i % FLAGS.display_step == 0 and batch_i > 0:
            val_loss = model.eval(sess,
                                  valid_sources_batch,
                                  valid_sources_lengths,
                                  valid_targets_batch,
                                  valid_targets_lengths)
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                  .format(i_epoch, config.num_epochs, batch_i,
                          len(train_source) // config.batch_size, train_loss, val_loss))
saver = tf.train.Saver()
saver.save(sess, FLAGS.model_path)
print('save model checkpoint to {}'.format(FLAGS.model_path))

