

import onnx
import torch 
from onnx2pytorch import ConvertModel

onnx_model_path = './weights/scrfd_10g_bnkps.onnx'

onnx_model = onnx.load(onnx_model_path)
pytorch_model = ConvertModel(onnx_model)
# Save the converted PyTorch model (optional)
torch.save(pytorch_model, 'scrfd_10g_bnkps.pth')