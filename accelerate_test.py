import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from accelerate import Accelerator, DeepSpeedPlugin

from monai.networks.nets.swin_unetr import SwinUNETR
from monai.losses import DiceLoss

class TestNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(TestNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.fc2 = nn.Linear(in_features=output_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    batch_size = 8
    dataset_size = 16
    input_data = torch.randn(dataset_size, *(2, 128, 128, 32))
    labels = torch.randn(dataset_size, *(1, 128, 128, 32))
    dataset = TensorDataset(input_data, labels)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    # model = TestNet(input_dim=input_dim, output_dim=output_dim)
    model = SwinUNETR(
        img_size = (128, 128, 32),
        in_channels = 2,
        out_channels = 1,
    )
    accelerator = Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = DiceLoss(sigmoid=True)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    for epoch in range(1000):
        model.train()
        for batch in dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
