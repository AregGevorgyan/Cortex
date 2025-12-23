# Cortex
Machine learning concepts implemented from scratch. I'm currently using numpy but may switch to JAX for some components.

Let me know if you have any suggestions or if you found this useful.

## Cortex Explained

I built this project during a 1 week informal hackathon at Startup Shell call Build-a-thon where it won the 'Scrappiest Project' award.

When building a ML framework from scratch there are a few levels of abstraction possible:
1. Low level opimized tensor operations like programing cuda kernals: I'm still learning systems but this would be a cool future project
2. Use an existing linear algebra library like Numpy and build a ML library ontop of it
3. Use an existing linear algebra library that had autograd and a few other features tailored to ML like JAX and build a neural network library ontop of it: there's a lot of cool libraries that I want to contribute to here

This project spesifically focuses on level 2.
It works by allowing for a `NeuralNetwork` object to be defined as a sequence of `Layer` objects. Each `Layer` object can be a a type of neural network layer like `Dense` or `Convolutional`, or a activation function like `ReLU`. Each `Layer` object has a forward method for calculating the output and a backward layer for the gradient (no autodiff yet). `NeuralNetwork` is also defined wiht a `Loss` object for the loss function and an `Optimizer` object for the optimization algorithm like `SGD` or `Adam`. 

## To run as notebook
```bash
uv run python -m ipykernel install --user --name cortex --display-name "Python (cortex)"
```
```bash
uv run jupyter notebook
```

## Some other cool projects
- https://github.com/MarioSieg/magnetron
- https://github.com/karpathy/micrograd
- https://github.com/tinygrad/tinygrad
- https://github.com/tanishqkumar/beyond-nanogpt
