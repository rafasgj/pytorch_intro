import torch

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.hidden_layer = torch.nn.Linear(2,2)
        self.output_layer = torch.nn.Linear(2,1)

    def forward(self, input):
        inner = self.sigmoid(self.hidden_layer(input))
        output = self.sigmoid(self.output_layer(inner))
        return output

    def train(
        self,
        train_data, train_tutor,
        epochs=5000, learning_rate=0.05,
        error_fn = torch.nn.MSELoss(), err_threshold=0.05,
        plot_every=10
    ):
        examples = torch.Tensor(train_data)
        expected = torch.Tensor(train_tutor).reshape(examples.shape[0], 1)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        MSE = float("+INF")
        all_losses = []
        current_loss = float("+INF")

        for epoch in range(epochs):
            epoch_mod = epoch % plot_every
            report_adjust = epoch_mod if epoch_mod > 0 else plot_every
            # remove current gradients
            optimizer.zero_grad()
            # Forward pass
            pred = self.forward(examples)
            loss = error_fn(pred, expected)
            MSE = loss.data.tolist()
            # Early stop
            if MSE <= err_threshold:
                break
            # Backward pass / Optimize
            loss.backward()
            optimizer.step()
            # Report
            current_loss += loss
            if epoch_mod == 0:
                all_losses.append(current_loss / report_adjust)
                current_loss = 0

        return MSE <= err_threshold, all_losses


p1 = MLP()

train_data = [
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
]

or_tutor = [0., 1., 1., 1.]  # OR
and_tutor = [0., 0., 0., 1.]  # AND
xor_tutor = [0., 1., 1., 0.]  # XOR

converged, all_losses = p1.train(train_data, xor_tutor, learning_rate=0.003, plot_every=1)

print("Converged" if converged else "Not converged")

import matplotlib.pyplot as plt
plt.plot([
    x.detach().numpy()
    for x in all_losses[1:]
])
plt.ylabel("Error")
plt.show()


for example in train_data:
    inp = torch.tensor(example)
    out = p1(inp)
    print(*example, round(out.data.tolist()[0]))
