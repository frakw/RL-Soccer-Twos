import torch
import torch.nn as nn

import numpy as np
import onnx

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.body_encoder = nn.Sequential(
            nn.Linear(336, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.branches = nn.ModuleList([
            nn.Linear(512, 3),
            nn.Linear(512, 3),
            nn.Linear(512, 3)
        ])

    def forward(self, x):
        x = self.body_encoder(x)
        outputs = [branch(x) for branch in self.branches]
        return outputs

# Load the ONNX model
onnx_model = onnx.load("./SoccerTwos.onnx")

# Define the name mapping
name_map = {
    "network_body._body_endoder.seq_layers.0.weight": "body_encoder.0.weight",
    "network_body._body_endoder.seq_layers.0.bias": "body_encoder.0.bias",
    "network_body._body_endoder.seq_layers.2.weight": "body_encoder.2.weight",
    "network_body._body_endoder.seq_layers.2.bias": "body_encoder.2.bias",
    "action_model._discrete_distribution.branches.0.weight": "branches.0.weight",
    "action_model._discrete_distribution.branches.0.bias": "branches.0.bias",
    "action_model._discrete_distribution.branches.1.weight": "branches.1.weight",
    "action_model._discrete_distribution.branches.1.bias": "branches.1.bias",
    "action_model._discrete_distribution.branches.2.weight": "branches.2.weight",
    "action_model._discrete_distribution.branches.2.bias": "branches.2.bias",
}

# Load the weights from the ONNX model
state_dict = {}
for initializer in onnx_model.graph.initializer:
    if initializer.data_type == 1:  # FLOAT
        data = np.frombuffer(initializer.raw_data, dtype=np.float32)
    elif initializer.data_type == 2:  # UINT8
        data = np.frombuffer(initializer.raw_data, dtype=np.uint8)
    # Add more elif conditions here for other data types if needed

    if data.dtype not in [np.float64, np.float32, np.float16, np.complex64, np.complex128, np.int64, np.int32, np.int16, np.int8, np.uint8, np.bool]:
        continue
    if initializer.name in name_map:
        state_dict[name_map[initializer.name]] = torch.from_numpy(
            data.reshape(tuple(dim for dim in initializer.dims))
        )

# Load the weights into the PyTorch model
pytorch_model = PyTorchModel()
pytorch_model.load_state_dict(state_dict)

print(pytorch_model)

torch.save(pytorch_model.state_dict(), "./SoccerTwos.pth")