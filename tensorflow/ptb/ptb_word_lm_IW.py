# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import six
from six.moves import xrange

import collections

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

#from tensorflow.models.rnn.ptb import reader
import reader

import solveAlphas as sa

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


def clip_squash_by_global_norm(t_list, clip_norm, name=None):

  if (not isinstance(t_list, collections.Sequence)
      or isinstance(t_list, six.string_types)):
    raise TypeError("t_list should be a sequence")

  t_list = list(t_list)

  use_norm = tf.global_norm(t_list, name)

  with ops.name_scope("clip_squash_by_global_norm") as name:

    # Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
    scale = tf.select(tf.less(use_norm, clip_norm),
                      0.0,
                      clip_norm)

    scale = tf.select(tf.equal(clip_norm, scale),
                      1 / use_norm,
                      scale)

    scale = clip_norm * scale

    values = [
        ops.convert_to_tensor(
            t.values if isinstance(t, ops.IndexedSlices) else t,
            name="t_%d" % i)
        if t is not None else t
        for i, t in enumerate(t_list)]

    values_clipped = []
    for i, v in enumerate(values):
      if v is None:
        values_clipped.append(None)
      else:
        with ops.colocate_with(v):
          values_clipped.append(
              array_ops.identity(v * scale, name="%s_%d" % (name, i)))

    list_clipped = [
        ops.IndexedSlices(c_v, t.indices, t.dense_shape)
        if isinstance(t, ops.IndexedSlices)
        else c_v
        for (c_v, t) in zip(values_clipped, t_list)]

    return list_clipped





class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):

    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    if is_training:
        self._input_data = tf.placeholder(tf.int32, [1, num_steps])
        self._targets = tf.placeholder(tf.int32, [1, num_steps])
    else:
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])


    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)

    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    if is_training:
        self._initial_state = cell.zero_state(1, data_type())
    else:
        self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # from tensorflow.models.rnn import rnn
    # inputs = [tf.squeeze(input_, [1])
    #           for input_ in tf.split(1, num_steps, inputs)]
    # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    if is_training:
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss)
    else:
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=data_type())])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)

    tvars = tf.trainable_variables()

    self.gradients, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                               config.max_grad_norm)

    #self.gradients = clip_squash_by_global_norm(tf.gradients(cost, tvars),
    #                                  config.max_grad_norm)

    self.gradients_size = len(self.gradients)

    self.placeholder_gradients = [tf.placeholder(tf.float32, shape=v.get_shape())
                                for v in tvars]

    optimizer = tf.train.GradientDescentOptimizer(self._lr)

    self._train_op = optimizer.apply_gradients(zip(self.placeholder_gradients, tvars))

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")

    self._lr_update = tf.assign(self._lr, self._new_lr)

    #Don't even think about it
    #with tf.device("/gpu:1"):
    #    self.mat = tf.placeholder(tf.float32, [self.batch_size, None])
    #    self.K = tf.matmul(self.mat, self.mat, transpose_b=True)


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000

class IWConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  p = 0.1
  C = 0.1
  max_grad_norm = 1.0
  num_layers = 1
  num_steps = 50
  # 1xh2048: 74532624
  # 2xh1500: 66022000
  # 2xh1000: 36018000
  hidden_size = 2048
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 1.0
  lr_decay = 0.9
  batch_size = 80
  vocab_size = 10000

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def run_epoch(session, model, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)
  for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                    model.num_steps)):
    #print("iters:", iters, end="\r", flush=True)
    fetches = [model.cost, model.final_state, eval_op]
    feed_dict = {}
    feed_dict[model.input_data] = x
    feed_dict[model.targets] = y
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    cost, state, _ = session.run(fetches, feed_dict)
    costs += cost
    iters += model.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * model.batch_size / (time.time() - start_time)), flush=True)

  #print(flush=True)
  return np.exp(costs / iters)



def run_epoch_train(session,
                    model,
                    data,
                    eval_op,
                    p,
                    C,
                    verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size,
                                                    model.num_steps)):

      losses = np.zeros(model.batch_size)
      gradients_list = []

      step_costs = 0.0

      for sample in xrange(model.batch_size):

        print("                           ", end="\r")
        print("sample: ", sample + 1, "step:", step + 1, end="\r", flush=True)

        #print("iters:", iters, end="\r", flush=True)
        fetches = [model.cost, model.final_state, model.gradients]
        feed_dict = {}
        feed_dict[model.input_data] = [x[sample]]
        feed_dict[model.targets] = [y[sample]]
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, state, gradients = session.run(fetches, feed_dict)
        step_costs += cost

        gradients_list.append(gradients)
        losses[sample] = cost

      costs += step_costs / model.batch_size
      iters += model.num_steps
      losses /= model.num_steps

      print(flush=True)

      print("build gradient matrix..", flush=True)

      for i, gradients in enumerate(gradients_list):

        print("                                                             ", end="\r", flush=True)

        print("gradient: ", i + 1, "   ", end="\r", flush=True)

        grads = [
            denseIndexedSlices(g).flatten() if isinstance(g, ops.IndexedSlicesValue) else g.flatten().astype(np.float32)
            for ig, g in enumerate(gradients)]


        #g = gradients[0]
        #grads_flatten = denseIndexedSlices(g).flatten() if isinstance(g, ops.IndexedSlicesValue) else #g.flatten().astype(np.float32)

        grads_flatten = np.concatenate(grads)

        # Allocate gradient matrix in the first step
        if i == 0:
            shape = (model.batch_size, grads_flatten.shape[0])
            print("gradient: ", i + 1, " allocating matrix ", shape, "..", end="\r", flush=True)
            # Force preallocating with ones, will be overwritten next
            grad_mat = np.ones(shape, np.float32)
            #self.grad_mat = np.zeros_like(self.grad_mat)
            print("                               ", end="\r", flush=True)

        # Fill gradient matrix
        grad_mat[i, :] = grads_flatten

      print(flush=True)

      print("solve alphas..", flush=True)

      # Solve for alphas
      #if step > 20: p = 0.1
      #if step > 30: p = 1.0 / step
      alphas = solveForAlphas(grad_mat,
                              p, # p
                              C, # C
                              losses)

      del(grad_mat)

      print("average gradients and weigh with alphas..", flush=True)

      # Average gradients and apply alphas
      #grads_alpha = self.average_gradients(gradients_bucket, alphas_bucket)

      grads_alphas = []
      for i, grad_batch in enumerate(gradients_list):

        print("gradient: ", i + 1, "    ", end="\r", flush=True)

        grads = [
                denseIndexedSlices(g) * alphas[i] if isinstance(g, ops.IndexedSlicesValue) else g.astype(np.float32) * alphas[i]
                for g in grad_batch]

        grads_alphas.append(grads)

      print(flush=True)
      print("averaging.. ", end="", flush=True)

      average_grads = np.mean(np.array(grads_alphas), 0)

      print("averaged", flush=True)

      del(grads_alphas)

      # Feed alpha scaled gradients into the TF graph and apply
      # them to the weights.

      print("applying gradients.. ", end="", flush=True)

      print("input/output feed.. ", end="", flush=True)

      # Input feed
      feed_dict = {}
      for i, grad in enumerate(average_grads):
        #print(bucket_id, i, self.placeholder_gradients[bucket_id][i].name, grad.shape)
        feed_dict[model.placeholder_gradients[i].name] = grad

      #fetches = [model.cost, model.final_state, eval_op]
      fetches = [eval_op]

      print("run.. ", end="", flush=True)

      session.run(fetches, feed_dict)

      print("finished", flush=True)

      #if verbose and (step * model.batch_size) % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(np.mean(losses)),
             iters * model.batch_size / (time.time() - start_time)), flush=True)

      #if verbose and ((step + 1)  % 25) == 0:
      #  valid_perplexity = run_epoch(session, model_valid, valid_data, tf.no_op())
      #  print("Step: %d Valid Perplexity: %.3f" % (step + 1, valid_perplexity))

      #if verbose and ((step + 1)  % 50) == 0:
      #  norm_add = 1.0
      #  new_norm = current_norm + norm_add
      #  new_norm = min(new_norm, 10.0)
      #  model.assign_max_global_norm(session, new_norm)
      #  current_norm = new_norm
      #  print("new max norm:", current_norm)

  #print(flush=True)
  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "iw":
    return IWConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)

# Returns np.float32
def solveForAlphas(grad_mat, p, C, losses):

    #print("gradient matrix shape:", grad_mat.shape, flush=True)

    print("K min max mean: ", end="", flush=True)

    K = np.matmul(grad_mat, np.transpose(grad_mat))
    #K = session.run(model.K, feed_dict={model.mat: grad_mat})

    print(np.amin(K), np.amax(K), np.mean(K), flush=True)

    #errors = np.log(np.array(bucket_loss) + 0.00001) / np.log(2.0) * p
    errors = losses * p # loss is cross entropy not perplexity

    print("losses min max mean :", min(losses), max(losses), np.mean(losses), flush=True)
    print("plosses min max mean:", min(errors), max(errors), np.mean(errors), flush=True)

    # Finally solve for alphas
    # Returns np.float32
    alphas = sa.solveAlphas(K, C, errors)

    print("alphas min max mean:", min(alphas), max(alphas), np.mean(alphas), flush=True)

    return(alphas)


# Deflates an IndexedSlices object
def denseIndexedSlices(slices):

    shape = slices.dense_shape

    dense_mat = np.zeros(shape, np.float32)

    if (len(shape) == 1):

        dense_mat[slices.indices] = slices.values

    elif (len(shape) == 2):

        for i in xrange(len(slices.indices)):

            dense_mat[slices.indices[i], :] = slices.values[i, :]

    else:

        raise ValueError("flattenIndexedSlices: Gradients with more than two dimensions not supported")

    return(dense_mat)



def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  print("read data.. ", end="", flush=True)
  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data
  print("finished", flush=True)

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:

    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):

      print("create model training ", end="", flush=True)
      m = PTBModel(is_training=True, config=config)

    with tf.variable_scope("model", reuse=True, initializer=initializer):

      print("validation ", end="", flush=True)
      mvalid = PTBModel(is_training=False, config=config)
      print("test ", end="", flush=True)
      mtest = PTBModel(is_training=False, config=eval_config)

    print("initialize variables..", flush=True)
    tf.initialize_all_variables().run()
    print(flush=True)

    for i in range(config.max_max_epoch):

      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)), flush=True)
      train_perplexity = run_epoch_train(session,
                                         m,
                                         train_data,
                                         m.train_op,
                                         config.p,
                                         config.C,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
