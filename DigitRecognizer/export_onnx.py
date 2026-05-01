import torch
from model import Net   

model = Net()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,   # works fine now
    do_constant_folding=True
)

print("ONNX Exported Successfully!")