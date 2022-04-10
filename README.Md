# Building a two-layer neural network classifier

A numpy implementation of ttwo-layer neural network classifier

## Installation

```
python=3.8.3
numpy=1.18.5
```

## Datasets

The data ser is from [**THE MNIST DATABASE of handwritten digits**](<http://yann.lecun.com/exdb/mnist/>)

```
Please put the data in:
./data
```



## Train & Test

```
# First, I trained the network using the training set of MNIST

python main.py

# Then, if you want to test the results of all parameters combinations

python test-all.py

# Or you want to test the results of specific parameters combination

python test-single.py --lr xx --hidden_neurons yy  --regularization zz

```

## Visualization

![](weight_0_visual.png.png)
