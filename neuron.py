import torch

# 2 input values, one output value, use bias
neuron = torch.nn.Linear(2, 1, bias=True)

# activation function
sigmoid = torch.nn.Sigmoid()

# activate neuron
values = [1, 0]
inp = torch.Tensor(values)
out = sigmoid(neuron(inp))

# print output
print(out.data.tolist())
