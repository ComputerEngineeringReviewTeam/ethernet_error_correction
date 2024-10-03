"""
TODO: CLEAN THIS
"""


import numpy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.etherbits import *

from src.ml.modules.network import Network
from src.ml.modules.autoencoder import Autoencoder
from src.ml.util.convert_tensors import round_float_to_bool_tensor

def frame_ones(f):
    return torch.count_nonzero(torch.count_nonzero(f, dim=1), dim=0)

path = "../../data/prep/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# trainSet = EtherBits(path, train=True, frame_size=1518, smallDataset=True)
# testSet = EtherBits(path, train=False, frame_size=1518, smallDataset=True)
trainSet = EtherBitsXor(path, "s100", train=True, frame_size=100)
testSet = EtherBitsXor(path, "s100", train=False, frame_size=100)
batch = 1000
print(len(trainSet))
print(len(testSet))
trainLoader = DataLoader(trainSet, batch_size=batch, shuffle=True)
testLoader = DataLoader(testSet, batch_size=batch, shuffle=True)

model = Network().to(device=device)

loss_fn = nn.modules.loss.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
model.train()
for epoch in range(20):
    for i, (x, y) in enumerate(trainLoader):
        x, y = x.to(device), y.to(device)

        predictions = model(x.to(torch.float32))
        #correct_y = (x ^ y)
        #loss = loss_fn(predictions, correct_y.to(torch.float32))
        loss = loss_fn(predictions, y.to(torch.float32))
        # loss = loss_fn((predictions * 2).to(torch.uint8).to(torch.bool).to(torch.float32), y.to(torch.float32))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch} - batch: {i}, Loss: {loss.item()}')
print("\nTraining finished")

test_loss, correct = 0, 0
all = 0
bits = 0
one_err = 0
all_err = 0
erred_frags = 0
corr_sl = 0
aaa = 0
og_bite, og_corr, og_one = 0, 0, 0
model.eval()
with torch.no_grad():
    for x, y in testLoader:

        x, y = x.to(device), y.to(device)
        pred = model(x.to(torch.float32))
        test_loss += loss_fn(pred, y.to(torch.float32)).item()
        # test_loss += loss_fn(pred, correct_y.to(torch.float32)).item()
        #print(torch.any(round_float_to_bool_tensor(pred), dim=1).sum())
        all += pred.shape[0]
        clean_y = x ^ y
        pred_b = (pred * 2).to(torch.uint8).to(torch.bool)

        diff_b = pred_b ^ y
        diff = torch.count_nonzero(pred_b ^ y, dim=1)
        bits += diff.sum().item()
        correct += diff.shape[0] - torch.count_nonzero(diff, dim=0)
        one_err += diff.shape[0] - torch.count_nonzero(diff - 1.0, dim=0)

        og_diff = torch.count_nonzero(x ^ y, dim=1)
        og_bite += og_diff.sum().item()
        og_corr += og_diff.shape[0] - torch.count_nonzero(og_diff, dim=0)
        og_one += og_diff.shape[0] - torch.count_nonzero(og_diff - 1.0, dim=0)


        all_err += torch.count_nonzero(x ^ y)
        erred_frags += torch.count_nonzero(torch.count_nonzero(y ^ x, dim=1), dim=0)
        slices_with_errors = torch.count_nonzero(y ^ x, dim=1)
        slices_with_errors_mask = slices_with_errors > 0
        diff_in_slices_with_errors = (pred_b ^ y)[slices_with_errors_mask]
        correct_slices_with_errors = diff_in_slices_with_errors.shape[0] - torch.count_nonzero(torch.count_nonzero(diff_in_slices_with_errors, dim=1), dim=0)
        corr_sl += correct_slices_with_errors.sum()
        aaa += (pred_b ^ y)[slices_with_errors_mask].shape[0]

print("avg loss: ", test_loss / len(testLoader))
print("avg bits predicted wrong: ", bits / (len(testLoader) * batch))
print("correct: ", correct.item(), "/", all, "=", correct.item() / all)
print("one err: ", one_err.item(), "/", all, "=", one_err.item() / all)
print("avg bit err og: ", all_err.item() / all)
print("corr og: ", og_corr.item())
print("og 1 err: ", og_one.item())
print()
print("all nr: ", all)
print("correct slices: ", corr_sl.item())
print("erred frags: ", erred_frags.item())
print("erred / all: ", erred_frags.item() / all)
print(corr_sl.item(), "/", erred_frags.item(), "=", corr_sl.item() / erred_frags.item())
print(aaa)
