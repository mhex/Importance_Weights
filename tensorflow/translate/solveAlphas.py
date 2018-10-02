import numpy as np

import collections

import six

from six.moves import xrange

from scipy.optimize import minimize

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def randAlphas(K, errors, n):
  alphas = np.random.rand(n)
  alphas = np.dot(K, alphas)
  alphas = alphas.astype(np.float32)
  #print("==========> alphas:")
  #print(errors)
  #print(n)
  #print(len(alphas))
  return alphas

def randAlphasTF(n):
  alphas = tf.random_normal(n)
  #print("K2")
  #print(K)
  #print("alphas:")
  #print(alphas)
  #alphas = tf.matmul(K, alphas, name="k_alphas_matmul")
  #alphas = tf.reshape(alphas, [-1], name="AlphasReshape")
  #alphas = tf.reduce_sum(alphas)
  return(alphas)

def solveAlphas(K, C, errors):
    
  #print(K.shape)
  #print(C)
  #print(errors)

  def objective(x, H, c):
    Hx = np.dot(H, x)
    xHx = np.dot(x, Hx)
    cx = np.dot(c, x)
    Qx = 0.5 * xHx - cx
    return Qx

  def gradient(x, H, c):
    Hx = np.dot(H, x)
    return Hx - c

  keep = np.diag(K) != 0.0
  alphas = np.zeros(keep.shape[0])
  if sum(keep) > 1.5:
    K_keep = K[keep][:,keep]
    norm = np.sqrt(np.diag(K_keep))
    norm = norm + np.mean(norm) * 0.01
    #print("norm okay")
    Hmat = K_keep / np.outer(norm, norm)
    #print("Hmat okay")
    #cvec = (-p * np.ones(K.shape[0]) * np.abs(error[keep])) / mynorm
    cvec = np.abs(errors[keep]) / norm
    #print("cvec okay")
    #lower = 0.0 * np.ones(Hmat.shape[0]) * mynorm
    lower = np.zeros(Hmat.shape[0])
    upper = C * norm
    #alphaNewOld=myoptim(Hmat, cvec, lower, upper)
    #import pdb
    #pdb.set_trace()
    res = minimize(objective,
		   0.5 * (lower + upper),
           args = (Hmat, cvec),
		   jac = gradient,
		   bounds = list(zip(lower,upper)),
		   method = 'TNC',
		   options = {'disp': False})
    alphasNew = res.x
    alphasNew = alphasNew / norm
    alphas[keep] = alphasNew
  elif sum(keep) > 0.5  and sum(keep) < 1.5:
    alphas[keep] = 1.0

  alphas = alphas.astype(np.float32)

  #print("alphas")
  #print(alphas)

  return alphas


###########
def get_alphas(grad_mat, losses):

  #grad_mat = tf.random_normal(shape=[num_buckets, 10000])

  with tf.device('/gpu:0'):
    K = tf.matmul(grad_mat, grad_mat, transpose_b=True, name="k_matmul_alphas")


  p = tf.constant(0.1)
  C = tf.constant(0.5)

  #losses = tf.pack(losses)

  #print("losses:")
  #print(losses)

  #losses = tf.reshape(losses, [-1])

  #print("r plosses:")
  #print(losses)

  #log_losses = tf.log(losses + 0.00001)

  #bs2 = tf.div(log_losses, tf.log(2.0))

  # error := log2(losses) * p
  errors = tf.mul(tf.div(tf.log(losses + 0.00001), tf.log(2.0)), p)
  #error = tf.mul(bs2, p) 
  #print("errors")
  #print(errors)

  #error = tf.zeros(tf.shape(error))

  # Returns a list of tensors with one element. Therefore get [0].
  alphas = tf.py_func(solveAlphas, [K, C, errors], [tf.float32])[0]
  #alphas = tf.py_func(randAlphas, [K, losses, tf.shape(K)[0]], [tf.float32])[0]

  return(alphas)


###############
def get_grad_vector(gradient_list, name=None):

  if (not isinstance(gradient_list, collections.Sequence)
      or isinstance(gradient_list, six.string_types)):
    raise TypeError("gradient_list should be a sequence")

  gradient_list = list(gradient_list)

  with ops.op_scope(gradient_list, name, "get_grad_vector") as name:

    values = [
        ops.convert_to_tensor(
            g.values if isinstance(g, ops.IndexedSlices) else g,
            name="g_%d" % i)
        if g is not None else g
        for i, g in enumerate(gradient_list)]

    #print("*** values")
    #for i, v in enumerate(values):
    #    print(i, v)
    #print()
    

    # Each 2-dim matrix into 1-dim vector
    #values = [
    #  tf.reshape(v, [-1])
    #  for v in values[gradient_index]]
    
    values = tf.reshape(values, [-1])

    #print("*** values reshape")
    #for i, v in enumerate(values):
    #    print(i, v)
    #print()

    # Concate all 1-dim vectors to one 1-dim vector dW
    values = tf.concat(0, values)

    #print("*** values concat")
    #print(values)


  return(values)



###########
def alphas_iw(gradients, losses, num_examples, name=None):

  #losses = [
  #    tf.reshape(l, [-1])
  #    for l in losses]

  #print("losses alphas")
  #print(losses)

  # list of bucket gradient vectors
  #grad_vectors = [
  grad_vectors = get_grad_vector(gradients, 0)
  #   for b in xrange(num_examples)]


  #print("*** grad_vectors")
  #print(grad_vectors)

  #grad_vectors = tf.concat(0, grad_vectors)

  #print("*** grad_vectors_concat")
  #print(grad_vectors)

  #grad_vectors = tf.reshape(grad_vectors, [-1])

  #print("*** grad_vectors_reshape")
  #print(grad_vectors)

  #grad_mat = grad_vectors

  grad_mat = tf.reshape(tf.transpose(grad_vectors), [256, -1], name="grad_vectors_reshape")

  #print("*** grad_mat")
  #print(grad_mat)

  #alphas = tf.ones(tf.shape(losses))
  alphas = get_alphas(grad_mat, losses)

  #alphas = tf.constant(0.01)
  #alphas = tf.reduce_sum(grad_vectors)
  #alphas =  tf.reduce_sum(tf.slice(grad_vectors[0], [0], [num_examples]))

 #alphas = tf.slice(grad_vectors, [0], [1])

  #alphas =  grad_vectors

  return(alphas)


###########
def scale_alpha_gradients(gradient_list, alphas, name=None):

  if (not isinstance(gradient_list, collections.Sequence)
      or isinstance(gradient_list, six.string_types)):
    raise TypeError("gradient_list should be a sequence")

  #gradient_list = list(gradient_list)

  with ops.op_scope(gradient_list, name, "iw_gradients") as name:

    values = [
        ops.convert_to_tensor(
            g.values if isinstance(g, ops.IndexedSlices) else g,
            name="g_%d" % i)
        if g is not None else g
        for i, g in enumerate(gradient_list)]

    iw_gradients = [
            array_ops.identity(tf.mul(v, alphas), name="%s_%d" % (name, i))
            if v is not None else None
            for i, v in enumerate(values)]

          #print(alphaGrads)

    list_iw_gradients = [
            ops.IndexedSlices(c_v, t.indices)
            if isinstance(t, ops.IndexedSlices)
            else c_v
            for (c_v, t) in zip(iw_gradients, gradient_list)]

  return(list_iw_gradients)



















