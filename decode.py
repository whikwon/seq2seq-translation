import numpy as np
import tensorflow as tf

import data_util
from config import Config
from model import Seq2Seq

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint_dir', './runs', 'checkpoint saved directory')
tf.flags.DEFINE_string('enc_vocab_path', './data/vocab.enc', 'directory encoder vocabulary exists')
tf.flags.DEFINE_string('dec_vocab_path', './data/vocab.dec', 'directory decoder vocabulary exists')

# model
config = Config()
sess = tf.Session()
model = Seq2Seq(config, mode='decode')
model.build()

# load saved model
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)
print('model restored from {}'.format(checkpoint_file))

# data load
vocab_enc = data_util.load_vocab(FLAGS.enc_vocab_path)[1]
vocab_dec = data_util.load_vocab(FLAGS.dec_vocab_path)[1]

# test
def chat():
    while True:
        test_input = data_util._get_user_input()
        test_input_id = data_util.sentence2id(vocab_enc, test_input)
        pred_ids = model.predict(
            sess, encoder_inputs=np.vstack([test_input_id for _ in range(config.batch_size)]),
            encoder_inputs_length=np.array([len(test_input_id) for _ in range(config.batch_size)]))[0]
        if not config.use_beamsearch_decode:
            reply = data_util.id2sentence(vocab_dec, pred_ids)
            print("German: {}".format(reply))
        else:
            pred_ids = pred_ids.T
            for pred in pred_ids:
                if config.end_token in pred:
                    end_id = np.where(pred == config.end_token)[0][0]
                    reply = data_util.id2sentence(vocab_dec, pred[:end_id])
                    print("German : {}".format(reply))


def main():
    chat()

if __name__ == '__main__':
    chat()