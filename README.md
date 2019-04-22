# tensorflow_mnist
Detecting handwritten digits using tensorflow

Initializer - xavier_initializer
Optimizer - AdamOptimizer

Nueral Network :
  1) Batch size = 100
  2) Layers = 4{
    a)hidden_layer_1 (512 neurons)
    b)hidden_layer_2 (256 neurons)
    c)hidden_layer_3 (128 neurons)
    d)output_layer (10 neurons)
  }
  3) Optimizer = AdamOptimizer
  4) Activation function = relu

Weights and biases of each layers are saved as:
(W1,W2,W3,WO).csv
(B1,B2,B3,BO).csv

Accuracy ~= 98%
