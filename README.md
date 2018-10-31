# Project Overview #

PaperbackWriter is a project that [Taylor Sasser](https://github.com/TaylorSasser) and I started in August 2018 with the goal of further understanding how neural networks can be used to replicate the English language. Furthermore, the creation of this project was made with the intent of making neural network code that was lightweight, efficient and intuitive.

# Requirements #

  * C++ 14
  * CUDA 9.2 or higher

# LSTM Overview #

LSTM networks function very similar to vanilla neural networks in that they utilize the same feedforward and backpropagation algorithms that traditional neural networks use. The key difference resides in the ability for LSTM networks to learn across sequences of data rather than the "one-to-one" model that traditional neural networks use (as seen in the image below). This phenomenon allows for LSTMs to establish long-term dependencies over patterns of data.

![](http://karpathy.github.io/assets/rnn/diags.jpeg)

The abstract model above can be more explicitly visualed using the diagram and equations shown below, which focus on one individual LSTM unit of a larger network. As modeled by the diagram, the input vector (x) is fed into the network at each timestep, is multiplied by the corresponding weight matrices and then bounded by the activation function. As a result two recurrent vectors are generated, the cell and hidden state, which allow for the network to encode data over sequences. 

![](https://cdn-images-1.medium.com/max/1600/0*LyfY3Mow9eCYlj7o.)

After the sequence has gone through a number of sequence steps, the change in the weights must be calculated w.r.t the loss function (a.k.a the gradient). This process of going back through the network and calculating the change of a cost function w.r.t each individual weight is called "backpropagation through time" (below). In the case of this project, the cross entropy function below was used to calculate the error, where the "p" is the ground truth and "q" is the network output. After "backpropagating" for a number of time steps the weights are incremented or decremented by a small amount according to the gradient and the optimizer used (ADAGrad, RMSProp, etc.).

![](https://i.imgur.com/7MOGDpG.png)

This process of "feeding" forward data and then "backpropagating through time" is repeated thousands or millions of times. As a result, after the training session, the network will be able to predict and generate its own sequences that closely models the data it was trained on.

# Example/Results #

Using the code from [kernel.cu](/kernel.cu) the network was trained with the following constants below: 
```
Learning Rate: 0.25
Weight Mean: 0
Weight Standard Deviation: 0.15
Hidden Size: 500
Input Size: 89
Sequence Length: 100
Optimizer: Adagrad 
```

The network was trained on the entire "Lord of the Rings" series (including the hobbit). After training for about 1.5 million iterations (1 epoch) the network produced the following graph based on error as a function of the number of iterations.
![](https://i.imgur.com/5XuDyUK.png)
Below are some of the resulting outputs after sampling the network.
```
ITERATION: 1070000

The Frodo past onder all hiten's on tree halt: Mhinis der and
l's air. We shadder Noint, Wow of Dorendo wroughs thoule betowest is gate in
here in has from be not op not out he could find longing
nom.'

'Dreying in not her. The bank no last, and muse get
were shorly and! 'We fell and and
Aragorn,' said Frodo they know even
the like the virnowned; Minilas,' said and his on thear Fanemares paid emwe, but in for now Bilds and will vanterngos, there before dright be of Goremened of the mounilad. Sogen in
even they had fielt if the him and his at thilt, would geew his sumble's the moved of the flokle,
```
