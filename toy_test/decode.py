import numpy as np
import tensorflow as tf

from model import Seq2Seq
import config
import data_util

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint_dir', './toy_runs', 'checkpoint saved directory')
tf.flags.DEFINE_string('source_file', '../data/letters_source.txt', 'path for source file')

# sample
source_sentences = data_util.read_file(FLAGS.source_file)
vocab = data_util.make_vocab(source_sentences)

# model loading
config = config.Config()
sess = tf.Session()
model = Seq2Seq(config, mode='decode')
model.build()

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)
print('model restored from {}'.format(checkpoint_file))

# test
def chat():
    while True:
        text_input = data_util._get_user_input()
        text = data_util.source2id(vocab, text_input)
        pred_ids = model.predict(
            sess, encoder_inputs=np.vstack([text for _ in range(config.batch_size)]),
            encoder_inputs_length=np.array([len(text) for _ in range(config.batch_size)]))[0]
        if not config.use_beamsearch_decode:
            reply = data_util.id2source(vocab, pred_ids)
            print("PREDICTION: {}".format(reply))
        else:
            pred_ids = pred_ids.T
            for pred in pred_ids:
                if config.end_token in pred:
                    print(pred)
                    end_id = np.where(pred == config.end_token)[0][0]
                    reply = data_util.id2source(vocab, pred[:end_id])
                    print("PREDICTION : {}".format(reply))


def main():
    chat()


if __name__ == '__main__':
    main()