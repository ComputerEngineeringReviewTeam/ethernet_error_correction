import torch
from torch import nn
from torch.utils.data import DataLoader

from src.ml.trainer import Trainer
from src.ml.datasets.EtherBits import EtherBits

training_data = EtherBits(
    directory="../../",
    train=True,
)

test_data = EtherBits(
    directory="../../",
    train=False,
)

batch_size = 100

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class TestNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits


model = TestNN().to(device)
print(model)

loss_fn = nn.HingeEmbeddingLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

trainer = Trainer(
    model=model,
    optimizer=optim,
    train_loader=train_dataloader,
    test_loader=test_dataloader,
    device=device
)

trainer.run(epochs=5, training_report_interval=100, test=True)
