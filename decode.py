import tensorflow as tf
from model import Seq2Seq
from config import Config

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint_dir', './runs', 'checkpoint saved directory')

# model
config = Config()
sess = tf.Session()
model = Seq2Seq(config, mode='decode')
model.build()

# load saved model
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)

# decode
text = [28, 5, 21, 21, 23, 0, 0]
logits = model.inference_logits
pred = sess.run(logits, feed_dict={model.encoder_inputs: [text] * config.batch_size,
                                   model.encoder_inputs_length: [len(text)] * config.batch_size})[1]
print(text)
print(pred)

