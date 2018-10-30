# Project Overview #

PaperbackWriter is a project that [Taylor Sasser](https://github.com/TaylorSasser) and I started in August 2017 with the goal of further understanding how neural networks can be used to replicate the English language. Furthermore, the creation of this project was made with the intent of making neural network code that was lightweight, efficient and intuitive.

# Requirements #

  * C++ 14
  * CUDA 9.2 or higher

# LSTM Overview #

LSTM networks function very similar to vanilla neural networks in that they utilize the same feedforward and backpropagation algorithms that vanilla neural networks use. The key difference resides in the ability for LSTM networks to learn across sequences of data rather than "one-to-one" relation as seen in the image below. As demonstrated by the image below LSTMs can utilize not only the data at a given sequence step but also data that was given to it in the past to predict the next vector in a sequence.

![](http://karpathy.github.io/assets/rnn/diags.jpeg)
