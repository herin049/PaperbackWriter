# Project Overview #

PaperbackWriter is a project that [Taylor Sasser](https://github.com/TaylorSasser) and I started in August 2018 with the goal of further understanding how neural networks can be used to replicate the English language. Furthermore, the creation of this project was made with the intent of making neural network code that was lightweight, efficient and intuitive.

# Requirements #

  * C++ 14
  * CUDA 9.2 or higher

# LSTM Overview #

LSTM networks function very similar to vanilla neural networks in that they utilize the same feedforward and backpropagation algorithms that vanilla neural networks use. The key difference resides in the ability for LSTM networks to learn across sequences of data rather than the "one-to-one" that traditional neural networks use as seen in the image below. This phenominon allows for LSTMs to establish long term dependencies over patterns of data.

![](http://karpathy.github.io/assets/rnn/diags.jpeg)

The abstract model above can be more explicitly visualed using the diagram and equations thorwn below, which focus on one individual LSTM unit of a larger network.

![](https://skymind.ai/images/wiki/greff_lstm_diagram.png)
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/2db2cba6a0d878e13932fa27ce6f3fb71ad99cf1)
