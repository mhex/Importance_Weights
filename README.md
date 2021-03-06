# Importance_Weights

Applying Importance Weights [1] to PTB Language Model [2], Seq2Seq Machine Translation [3],
and Neural GPU [4] 

The "importance weight" method (IW) speeds up learning of "difficult" data sets including unbalanced data, highly non-linear data, or long-term dependencies in sequences. An importance weight is assigned to every training data point and controls its contribution to the total weight update. The importance weights are obtained by solving a quadratic optimization problem and determines the learning informativeness of a data point.

[1]
Hochreiter & Obermayer. Optimal gradient-based learning using importance weights
Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005.
https://ieeexplore.ieee.org/document/1555815

[2]
https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb

[3]
Original code from Google is not available anymore
Sutskever, Vinyals, Le. Sequence to sequence learning with neural networks,
NIPS'14 Proceedings

[4]
https://github.com/tensorflow/models/tree/master/research/neural_gpu
