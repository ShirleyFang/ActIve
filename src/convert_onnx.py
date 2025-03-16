import os
import torch
import onnx
from contributing_factors_analysis.lifestyle_factors_train_model import LifestyleNN  # 确保 lifestyle_factors_train_model.py 在 src 目录下

# 1️⃣ 获取当前 `convert_onnx.py` 的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2️⃣ 计算 `models/lifestyle_nn.pth` 和 `models/lifestyle_nn.onnx` 的路径
model_path = os.path.join(current_dir, "..", "models", "lifestyle_nn.pth")
onnx_path = os.path.join(current_dir, "..", "models", "lifestyle_nn.onnx")

# 3️⃣ 确保路径在 Windows 和 macOS/Linux 下都能用
model_path = os.path.normpath(model_path)
onnx_path = os.path.normpath(onnx_path)

# 4️⃣ 确保模型文件存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 模型文件未找到: {model_path}")

# 5️⃣ 加载 PyTorch 训练好的模型
input_size = 6  # ⚠️ 根据 `analyze_lifestyle()` 方法中的 `input_size=6`
num_classes = 4 

model = LifestyleNN(input_size=input_size, num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置为推理模式

# 6️⃣ 生成示例输入
dummy_input = torch.randn(1, input_size)  # 确保 batch_size = 1

# 7️⃣ 导出为 ONNX
torch.onnx.export(
    model, dummy_input, onnx_path,
    export_params=True, opset_version=11,
    do_constant_folding=True,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ PyTorch 模型已成功转换为 ONNX: {onnx_path}")

