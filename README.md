Basic perceptron implementation with iris dataset.

## Neural Network

There are many kinds of models for machine learning. Neural Network is one of them. It draws inspiration from the human brain and its biological neural networks. Neural network imitates the learning, memorizing, and generalizing ability of brain. The learning operation of a neural network occurs using training data. There are some basic points of it such as input values, output values and rules. The mathematical model of neural system of the human brains represented in the following figure. 

![alt text](https://github.com/erkanfatma/BasicPerceptron/blob/main/img/Picture1.png)

Neural network is a multi-layer system. The first layer is called as input layer and the last layer is named as output layer. Other layers called as hidden layers.

![alt text](https://github.com/erkanfatma/BasicPerceptron/blob/main/img/Picture2.png)

Every layer contains a certain number of neurons. These neurons are connected to each other by synapse. Every synapse has a weight value, and those weight value represent the importance of the information of that neuron.
A neural consists of some parts:
1. Inputs
Inputs are a data that comes to the neurons. The data from these inputs sent to the neurons to be collected.
2. Weights
The information that came from the neurons is multiplied by the weight of the connections before reaching the other neuron through the inputs and transmitted to the neurons. With this way, effects of the input on the output will be modified.
3. Function
A function that takes the weighted sum of all inputs in the previous layer and then generates an output value and passes it to the next layer.
4. Outputs
Values that comes out of the function is the output value.

## Perceptrons

Perceptron is the first generation and basic building block of neural networks and it is a simply computational model. It is a single-layer neural network. It helps to solve not complex problems. The perceptron model contains only two models: input layer and output layer, there are no hidden layers. Inputs are used to calculate the weighted input for each node. This model uses activation function for classification.
The perceptron consists of four parts: input values, weight & bias, net sum, and activation function.

#### How Perceptron Works?

![alt text](https://github.com/erkanfatma/BasicPerceptron/blob/main/img/Picture3.png)

1. Let call inputs as x. These inputs are multiplied with the weights represented as w. Lets call that k.
![alt text](https://github.com/erkanfatma/BasicPerceptron/blob/main/img/Picture4.png)

2. Add all multiplied values and it is called as weighted sum.

![alt text](https://github.com/erkanfatma/BasicPerceptron/blob/main/img/Picture6.gif)

3. Apply weighted sum to the correct Activation function.

Weights are used to show strength of that node. Bias value is used to shift the activation curve down or up. Activation function used to map input with required values such as (0,1).
In summary, perceptron is used to classify data into two parts. For instance, we have an image, and we find that if the image is bird or not. Another saying, we have n different images, and those images are classified as 0 and 1.
 
