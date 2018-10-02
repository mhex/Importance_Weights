# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from tensorflow.models.rnn.translate import data_utils

import solveAlphas as sa

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, sess, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    #self.learning_rate_set_op = self.learning_rate.assign(1.0)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w = tf.get_variable("proj_w", [size, self.target_vocab_size], dtype=tf.float32)
      w_t = tf.transpose(w)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=tf.float32)
      output_projection = (w, b)

      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        #return tf.cast(
        return tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                       num_samples, self.target_vocab_size)
        #    tf.float32)

      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.nn.rnn_cell.GRUCell(size)
    if use_lstm:
      single_cell = tf.nn.rnn_cell.BasicLSTMCell(size)
    cell = single_cell
    if num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return tf.nn.seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=do_decode,
          dtype=tf.float32)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function, per_example_loss=False)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
      self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function, per_example_loss=False)


    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()

    if not forward_only:

      self.gradient_norms = []
      self.updates = []
      self.gradients_bucket = []
      self.mat = [] # gradient matrix
      self.K = [] # gradient kernel

      #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      opt = tf.train.GradientDescentOptimizer(1.0)
      #opt = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-4)


      self.placeholder_gradients = [tf.placeholder(tf.float32, shape=v.get_shape())
                                for v in params]

      for b in xrange(len(buckets)):

        gradients, norm = tf.clip_by_global_norm(tf.gradients(self.losses[b], params),
                                                         max_gradient_norm)

        self.gradients_size = len(gradients)

        self.gradients_bucket.append(gradients)

        self.gradient_norms.append(norm)

        self.updates.append(opt.apply_gradients(
            zip(self.placeholder_gradients, params), global_step=self.global_step))

        #with tf.device("/gpu:1"):
        #    self.mat.append(tf.placeholder(tf.float32, [self.batch_size, None]))
        #    self.K.append(tf.matmul(self.mat[b], self.mat[b], transpose_b=True))

    self.saver = tf.train.Saver(tf.all_variables())


  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, C, p, forward_only, global_step):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """

    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    if not forward_only:

        # Forward and backward/prop step

        sentence_nr = 1

        # List of losses for each sample
        current_losses = []

        # List of gradient lists for each sample
        gradients_bucket = []

        # For each sentence in batch retrieve gradients
        for sentence in xrange(self.batch_size):

            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(self.train_buckets_scale))
                            if self.train_buckets_scale[i] > random_number_01])

            print("sentence: ", sentence_nr, "bucket: ", bucket_id, "     ", end="\r", flush=True)

            encoder_size, decoder_size = self.buckets[bucket_id]

            # Batch of 1
            encoder_inputs, decoder_inputs, target_weights = self.get_batch(
                self.train_set, bucket_id, 1)

            # Build output feed only for the losses and gradients
            # No gradient updates yet
            output_feed = [self.losses[bucket_id], self.gradient_norms[bucket_id]]  # Loss for this sentence
            for g in xrange(self.gradients_size):  # gradients for sentence
                output_feed.append(self.gradients_bucket[bucket_id][g])

            # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
            input_feed = {}
            for l in xrange(encoder_size):
                input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
            for l in xrange(decoder_size):
                input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
                input_feed[self.target_weights[l].name] = target_weights[l]

            # Since our targets are decoder inputs shifted by one, we need one more.
            last_target = self.decoder_inputs[decoder_size].name
            input_feed[last_target] = [0]

            # RUN DOS RUN
            outputs = session.run(output_feed, input_feed)

            current_loss = outputs[0]

            # Norm scalar from gradient clipping/scaling
            norm = outputs[1]

            gradients = outputs[2:]

            # Extend list of gradient lists over samples/sentences
            gradients_bucket.append(gradients)

            # Extend list of losses over samples/sentences
            current_losses.append(current_loss)

            sentence_nr += 1

        print()

        # Calculate alphas and avg. loss

        # Build gradient dense matrix

        print("build gradient matrix..", flush=True)

        # For minibatch 200 and 2 layer a 256 GRUs shape of
        # grad_mat is (200, 32795712)

        for i, gradient_bucket in enumerate(gradients_bucket):

            print("                                                  ", end="\r", flush=True)

            print("gradient: ", i + 1, "   ", end="\r", flush=True)

            grads = [
                self.denseIndexedSlices(g).flatten() if isinstance(g, ops.IndexedSlicesValue) else g.flatten().astype(np.float32)
                #g.flatten()
                for g in gradient_bucket]

            grads_flatten = np.concatenate(grads)

            # Allocate gradient matrix in the first step
            #if self.first_step and (i == 0):
            if i == 0:
                shape = (self.batch_size, grads_flatten.shape[0])
                print("gradient: ", i + 1, " allocating matrix ", shape, "..", end="\r", flush=True)
                # Force preallocating with ones, will be overwritten next
                grad_mat = np.ones(shape, np.float32)
                print("                               ", end="\r", flush=True)

            # Fill gradient matrix
            grad_mat[i, :] = grads_flatten

        print(flush=True)

        print("solve alphas..", flush=True)

        # Solve for alphas

        p = p * self.learning_rate.eval()
        alphas = self.solveForAlphas(session,
                                     grad_mat,
                                     p, # p
                                     C, # C
                                     np.array(current_losses),
                                     bucket_id).astype(np.float32)

        del(grad_mat)

        print("average gradients and weigh with alphas..", flush=True)

        # Average gradients and apply alphas
        #grads_alpha = self.average_gradients(gradients_bucket, alphas_bucket)

        grads_alphas = []
        for i, grad_batch in enumerate(gradients_bucket):

            print("gradient: ", i + 1, "    ", end="\r", flush=True)

            grads = [
                self.denseIndexedSlices(g) * alphas[i]
                if isinstance(g, ops.IndexedSlicesValue)
                else g.astype(np.float32) * alphas[i]
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
        input_feed = {}
        for i, grad in enumerate(average_grads):
            #print(bucket_id, i, self.placeholder_gradients[bucket_id][i].name, grad.shape)
            input_feed[self.placeholder_gradients[i].name] = grad

        # Output feed
        output_feed = [self.updates[bucket_id]] # gradient update

        print("run.. ", end="", flush=True)

        # RUN DOS RUN
        outputs = session.run(output_feed, input_feed)

        print("finished", flush=True)

        # Average loss over the batch
        loss_avg = np.mean(np.array(current_losses))

        return norm, loss_avg, None  # Gradient norm, loss, no outputs.

    else:

        # Forward only step

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name

        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        output_feed = [self.losses[bucket_id]]  # Loss for this batch.
        for l in xrange(decoder_size):  # Output logits.
            output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)

        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def solveForAlphas(self, session, grad_mat, p, C, errors, bucket_id):

        #print("gradient matrix shape:", grad_mat.shape, flush=True)

        print("K min max mean: ", end="", flush=True)

        K = np.matmul(grad_mat, np.transpose(grad_mat))
        #K = session.run(self.K[bucket_id], feed_dict={self.mat[bucket_id]: grad_mat})

        print(np.amin(K), np.amax(K), np.mean(K), flush=True)

        perrors = errors * p # loss is cross entropy not perplexity

        print("errors min max mean:", min(errors), max(errors), np.mean(errors), flush=True)
        print("perrors min max mean:", min(perrors), max(perrors), np.mean(perrors), flush=True)

        # Finally solve for alphas
        alphas = sa.solveAlphas(K, C, perrors)

        print("alphas min max mean:", min(alphas), max(alphas), np.mean(alphas), flush=True)

        return(alphas)

  # Not used
  def average_gradients(self, grads_batch, alphas):
    """Calculate the average gradient weighted by alphas
    Args:
        grads_batch: List of lists of gradients. The outer list
        is over batch elements. The inner list is over the
        individual gradients.
    Returns:
        List gradients where the gradient has been averaged
        across all batch elements weighted by alphas.
    """
    grads_batch_alphas = []
    for i, grad_batch in enumerate(grads_batch):
        #grads = [
        #    g.values if isinstance(g, ops.IndexedSlices) else g
        #    if g is not None else g
        #    for g in grad_batch]

        print("gradient: ", i + 1, "    ", end="\r", flush=True)

        grads = [
            self.denseIndexedSlices(g) * alphas[i] if isinstance(g, ops.IndexedSlicesValue) else g * alphas[i]
            for g in grad_batch]

        grads_batch_alphas.append(grads)

    print(flush=True)
    print("averaging..", end="", flush=True)

    average_grads = np.mean(np.array(grads_batch_alphas), 0)

    print(" averaged", flush=True)

    return average_grads



  # Deflates an IndexedSlices object
  def denseIndexedSlices(self, slices):

      shape = slices.dense_shape

      dense_mat = np.zeros(shape)

      if (len(shape) == 1):

            dense_mat[slices.indices] = slices.values

      elif (len(shape) == 2):

        for i in xrange(len(slices.indices)):

            dense_mat[slices.indices[i], :] = slices.values[i, :]

      else:

        raise ValueError("flattenIndexedSlices: Gradients with more than two dimensions not supported")

      return(dense_mat.astype(np.float32))


  def get_batch(self, data, bucket_id, batch_size):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

  ############################################
  # Importance Weights
  ############################################

