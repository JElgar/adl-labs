from typing import Callable

import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn import datasets


feature_count = 4
hidden_layer_size = 100
class_count = 3


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layer_size: int,
        output_size: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        return x


def accuracy(probs: torch.FloatTensor, targets: torch.Tensor) -> float:
    """
    Args:
        probs: A float32 tensor of shape ``(batch_size, class_count)`` where each value
            at index ``i`` in a row represents the score of class ``i``.
        targets: A long tensor of shape ``(batch_size,)`` containing the batch examples'
            labels.
    """
    number_of_correct_predictions = (probs.argmax(axis=1) == targets).sum()
    return float(number_of_correct_predictions) / targets.shape[0]
    # return sum((prob.argmax() == target) for prob, target in zip(probs, targets)) / len(
    #     targets
    # )


# Define the model
model = MLP(feature_count, hidden_layer_size, class_count)

iris = datasets.load_iris()
labels = iris["target"]
preprocessed_features = (iris["data"] - iris["data"].mean(axis=0)) / iris["data"].std(
    axis=0
)
train_features, test_features, train_labels, test_labels = train_test_split(
    preprocessed_features, labels, test_size=1 / 3
)

features = {
    "train": torch.tensor(train_features, dtype=torch.float32),
    "test": torch.tensor(test_features, dtype=torch.float32),
}
labels = {
    "train": torch.tensor(train_labels, dtype=torch.long),
    "test": torch.tensor(test_labels, dtype=torch.long),
}

device = torch.device("cuda")
model.to(device)
features["train"] = features["train"].to(device)
features["test"] = features["test"].to(device)
labels["train"] = labels["train"].to(device)
labels["test"] = labels["test"].to(device)

# The optimizer we'll use to update the model paramters
optimizer = optim.SGD(model.parameters(), lr=0.05)

# The lostt function
criterion = nn.CrossEntropyLoss()

# Run epochs
for epoch in range(100):
    # Compute forward pass
    logits = model.forward(features["train"])
    # Calculate the loss
    loss = criterion(logits, labels["train"])

    print(
        f"epoch: {epoch}, train accuracy: {accuracy(logits, labels['train']):.2f}, loss: {loss.item():5f}"
    )

    # Compute the backward pass (populdate the .grad attributes for each parameter)
    loss.backward()

    # Compute new parameters using the gradients
    optimizer.step()

    # Clear out the grad calculated in this step
    optimizer.zero_grad()

logits = model.forward(features["test"])
test_accuracy = accuracy(logits, labels["test"]) * 100
print(f"Test Accuracy: {test_accuracy:2.2f}")
