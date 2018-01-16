import os
import data_util
import tensorflow as tf
import config
from model import Seq2Seq

FLAGS = tf.flags.FLAGS
# model
tf.flags.DEFINE_string('model_dir', './toy_runs', 'directory for trained model')
tf.flags.DEFINE_string('model_path', './toy_runs/model.ckpt', 'path for saving trained parameter')
tf.flags.DEFINE_string('source_file', '../data/letters_source.txt', 'path for source file')
tf.flags.DEFINE_integer('print_every', 20, 'steps for displaying status of training')

# checkpoint dir check
if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

# config loading
config = config.Config()

# data loading
train_source, train_target, valid_source, valid_target = data_util.load_data(FLAGS.source_file)
batch_generator = data_util.get_batches(
    train_source, train_target, config.num_epochs, config.batch_size)
val_generator = data_util.get_batches(valid_source, valid_target, 500, config.batch_size)

# model loading
model = Seq2Seq(config, mode='train')
model.build()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for batch in batch_generator:
    train_loss, step, summary = model.train(sess, *batch)
    if step % FLAGS.print_every == 0:
        val_loss = model.eval(sess, *next(val_generator))
        print('Iterations: {:>4} --- Train loss: {:>6.3f} --- Val loss: {:>6.3f}'
              .format(step, train_loss, val_loss))
saver = tf.train.Saver()
saver.save(sess, FLAGS.model_path)
print('save model checkpoint to {}'.format(FLAGS.model_path))