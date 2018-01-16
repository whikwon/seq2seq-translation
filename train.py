import os

import tensorflow as tf

from config import Config
from data_util import load_data, get_batch
from model import Seq2Seq

FLAGS = tf.flags.FLAGS

# model
tf.flags.DEFINE_string('model_dir', './runs', 'directory for trained model')
tf.flags.DEFINE_string('summary_dir', './runs', 'directory for model summary')
tf.flags.DEFINE_string('model_path', './runs/model', 'path for saving trained parameter')
tf.flags.DEFINE_integer('print_every', 50, 'steps for displaying status of training')
tf.flags.DEFINE_integer('num_checkpoints', 5, 'Maximum # holding checkpoints')
tf.flags.DEFINE_integer('checkpoint_every', 2000, 'steps for saving checkpoint')
tf.flags.DEFINE_string('train_source_file', 'train_ids.enc', 'filename for train source')
tf.flags.DEFINE_string('train_target_file', 'train_ids.dec', 'filename for train target')
tf.flags.DEFINE_string('test_source_file', 'test_ids.enc', 'filename for test source')
tf.flags.DEFINE_string('test_target_file', 'test_ids.dec', 'filename for test target')
tf.flags.DEFINE_boolean('keep_training', False, 'whether restoring a model for training or not')

# checkpoint dir check
if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

# config loading
config = Config()

# data loading
train_source, train_source_length, \
    train_target, train_target_length = load_data(
        FLAGS.train_source_file, FLAGS.train_target_file,
        config.source_max, config.target_max)

test_source, test_source_length, \
    test_target, test_target_length = load_data(
        FLAGS.test_source_file, FLAGS.test_target_file,
        config.source_max, config.target_max)

batch_generator = get_batch(
    train_source, train_source_length, train_target,
    train_target_length, config.batch_size, config.num_epochs)

# model build
model = Seq2Seq(config, mode='train')
model.build()
sess = tf.Session()
if FLAGS.keep_training:
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.model_dir)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)
else:
    sess.run(tf.global_variables_initializer())

# summaries & saver
summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)
saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

# training
while batch_generator:
    train_loss, step, summary = model.train(sess, *next(batch_generator))
    summary_writer.add_summary(summary, step)
    if step % FLAGS.print_every == 0:
        print('Iterations: {:>4} --- Loss: {:>6.3f}'.format(step, train_loss))
        if step % FLAGS.checkpoint_every == 0:
            saver.save(sess, FLAGS.model_path, global_step=step)
            print('save model checkpoint to {}'.format(FLAGS.model_path))

saver = tf.train.Saver()
saver.save(sess, FLAGS.model_path)
print('save model checkpoint to {}'.format(FLAGS.model_path))

