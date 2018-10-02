import numpy as np
import solveAlphas as sa

import tensorflow as tf

M = tf.random_normal(shape=[45, 50])

K = tf.matmul(M, M, transpose_b=True)

      #K = tf.to_float(K)

p = tf.constant(0.1)
C = tf.constant(0.1)
      #two = tf.constant(2.0)

      #ploss = tf.pack(self.losses)

      #print("plosses:")
      #print(ploss)

      #ploss = tf.reshape(ploss, [-1])

      #print("r plosses:")
      #print(ploss)

      #lloss = tf.log(ploss)
      #bs2 = tf.div(lloss, tf.log(two))
      #error = tf.mul(bs2, p)

#error = tf.ones([5,1])
error = tf.random_normal(shape=[45])

print(K)
print(error)

#K = tf.matmul(K, error)

      #lb = tf.constant(45, dtype=tf.int32)
      # Returns a list of tensors with one element. Therefore get [0].
alphas = tf.py_func(sa.solveAlphas, [K, p, C, error], [tf.float32])[0]

#alphas = tf.slice(alphas, [0], [5])

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

sess.run(alphas)

print(alphas.eval(session=sess))



