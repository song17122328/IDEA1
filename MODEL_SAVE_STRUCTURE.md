# 模型保存结构说明

## 目录结构

运行剪枝脚本后，会在 `prune_log/` 目录下生成以下结构：

```
prune_log/
└── llama_prune/                          # 主日志目录 (由 --save_ckpt_log_name 参数指定)
    ├── description.txt                   # 主配置文件（记录所有运行参数）
    ├── pytorch_model.bin                 # ⭐ 最终保存的剪枝后模型（这是最重要的文件）
    │
    ├── 2025-11-12-10-48-13/              # 第一次运行的时间戳子目录
    │   ├── description.txt               # 本次运行的配置参数
    │   ├── train.sh                      # 本次运行的完整命令
    │   ├── training.log                  # 详细的训练日志
    │   ├── pytorch_model.bin             # 本次运行的checkpoint
    │   └── latest_model.bin              # 最新的模型快照
    │
    ├── 2025-11-12-11-30-25/              # 第二次运行的时间戳子目录
    │   ├── description.txt
    │   ├── train.sh
    │   ├── training.log
    │   └── ...
    │
    └── ...                               # 更多运行记录
```

## 关键文件说明

### 1. `prune_log/llama_prune/pytorch_model.bin`
- **最重要的文件**：这是最终保存的剪枝后模型
- 每次运行都会覆盖这个文件
- 包含：
  - 剪枝后的完整模型（`model`）
  - Tokenizer（`tokenizer`）

### 2. 时间戳子目录（如 `2025-11-12-10-48-13/`）
- 每次运行都会创建新的时间戳目录
- 用于追溯历史运行记录
- 包含该次运行的所有详细信息

## 模型大小验证

### 方法 1：使用提供的检查脚本

我们提供了一个专门的脚本来检查模型大小：

```bash
# 检查所有剪枝后的模型
python check_model_size.py

# 对比原始模型和剪枝后模型
python check_model_size.py \
    --original_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruned_model prune_log/llama_prune/pytorch_model.bin
```

### 方法 2：手动检查文件大小

```bash
# 查看所有模型文件大小
du -h prune_log/llama_prune/*.bin

# 对比原始模型目录和剪枝后模型
du -h /newdata/LLMs/Llama-3-8B-Instruct/
du -h prune_log/llama_prune/pytorch_model.bin
```

### 方法 3：Python 脚本快速检查

```python
import torch
import os

# 检查文件大小
pruned_path = 'prune_log/llama_prune/pytorch_model.bin'
file_size_mb = os.path.getsize(pruned_path) / (1024 * 1024)
print(f"剪枝后模型文件大小: {file_size_mb:.2f} MB")

# 加载模型并检查参数数量
checkpoint = torch.load(pruned_path, map_location='cpu')
model = checkpoint['model']

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数数量: {total_params:,}")
print(f"理论大小 (FP16): {total_params * 2 / (1024**3):.2f} GB")
print(f"理论大小 (FP32): {total_params * 4 / (1024**3):.2f} GB")
```

## 为什么物理占用可能没有明显减少？

即使参数数量减少了 16.22%，物理文件大小可能不会有同等比例的减少，原因如下：

1. **模型结构元数据**
   - 保存的不只是权重，还包括模型结构、配置等
   - Tokenizer 也占用一定空间

2. **PyTorch 序列化开销**
   - `.bin` 文件包含额外的序列化信息
   - 使用 pickle 格式会有额外开销

3. **精度格式**
   - 原始模型可能使用了不同的数值精度
   - 建议检查是否都是 FP16 格式

4. **实际参数减少验证**
   ```python
   # 从日志可以看到：
   # 剪枝前: 8,030,261,248 参数
   # 剪枝后: 6,727,929,856 参数
   # 减少: 1,302,331,392 参数 (16.22%)

   # 理论文件大小减少（FP16）:
   # 1,302,331,392 * 2 bytes = 2.43 GB
   ```

## 加载剪枝后的模型

```python
import torch
from transformers import AutoTokenizer

# 加载剪枝后的模型
checkpoint = torch.load('prune_log/llama_prune/pytorch_model.bin',
                       map_location='cuda:0')

model = checkpoint['model']
tokenizer = checkpoint['tokenizer']

# 使用模型
model.eval()
input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0]))
```

## 推荐的工作流程

1. **运行剪枝**
   ```bash
   python llama3.py --pruning_ratio 0.25 \
       --base_model /path/to/model \
       --save_model \
       ...其他参数
   ```

2. **验证模型大小**
   ```bash
   python check_model_size.py \
       --original_model /path/to/original/model \
       --pruned_model prune_log/llama_prune/pytorch_model.bin
   ```

3. **检查日志**
   ```bash
   # 查看最新运行的日志
   cat prune_log/llama_prune/$(ls -t prune_log/llama_prune/ | grep "^20" | head -1)/training.log
   ```

4. **备份重要模型**
   ```bash
   # 如果对结果满意，建议备份
   cp prune_log/llama_prune/pytorch_model.bin ./llama3_pruned_25pct.bin
   ```

## 注意事项

- `pytorch_model.bin` 会在每次运行时被覆盖
- 如果需要保留多个版本，请在运行后立即备份
- 时间戳子目录会一直保留，不会被覆盖
- 建议定期清理旧的时间戳目录以节省空间
