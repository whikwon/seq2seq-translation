# Seq2seq-translation
Tensorflow implementation of a paper [Neural Machine Translsation By Joint Learning To Align and Translate](https://arxiv.org/abs/1409.0473), Bahdanau et al. ICLR 2015. 

### toy-example
- Dataset: [sequence of chars](https://github.com/udacity/deep-learning/blob/master/seq2seq/data/letters_source.txt)
- Training purpose: reversing the sequence of characters
```
cd toy_test 
python train.py 
python decode.py
```
### English-German translation
- Dataset: [English-German pair sentences](https://github.com/harvardnlp/seq2seq-attn/data)
- Training purpose: translation
```
python data_util.py
python train.py
python decode.py
```

### References
- [Neural Machine Translation Tutorial](https://github.com/tensorflow/nmt)
- [TF-seq2seq](https://github.com/JayParks/tf-seq2seq)
- [Udacity - seq2seq](https://github.com/udacity/deep-learning/tree/master/seq2seq)
- [Stanford Univ. Tensorflow tutorial - Chatbot](https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/2017/assignments/chatbot)

### License
MIT
