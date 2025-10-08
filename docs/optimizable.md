# Selective Parameter Freezing

MFNetJax model nodes support an `optimizable` flag that lets you freeze or unfreeze their parameters during training.

## How It Works

- By default, all nodes are optimizable.
- Call `model.set_optimizable(False)` on a node to freeze its parameters.
- `.fit()` will zero out gradients for any frozen nodes before applying updates.
- This enables staged or targeted training of subgraphs.

## Example

```python
# Assume m1, m2, m3 are node functions in your MFNetJax graph
m1.set_optimizable(False)
m2.set_optimizable(False)
m3.set_optimizable(True)

# Only node 3 will update during training
mfnet.fit(train_data, n_iters=1000)
```
