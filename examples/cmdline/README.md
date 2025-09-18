This contains examples of how to run the mfnets cmdline interface. You can run any example by e.g., running

```
make ex1
```

This will create a new directory in `results/ex1` with the results for the first example. Below is a brief description of the examples. The training data is located in `sample_data/single_output` and `sample_data/multi_output`. You can check those files for the exact training/testing inputs and outputs. The values there are chosen to be fairly random, be warned.

- Example 1:
  + SISO
  + Regression 
  + linear scale-shift

- Example 2:
  + SISO
  + Bayesian
  + linear scale-shift
  
- Example 3:
  + SIMO
  + Regression
  + linear scale-shift
  
- Example 4:
  + Two individual single fidelities
  + SISO
  + Regression
  + linear scale-shift
  
- Example 5:
  + SIMO
  + Regression
  + linear and feedforward

- Example 6:
  + SISO
  + Regression
  + linear and feedforward, with model average edge
  
- Example 7:
  + Single fidelity
  + SISO
  + Regression
  + Neural network

- Example 8:
  + SIMO 
  + Regression
  + Polynomial + Polynomial-Linear Scale shift
