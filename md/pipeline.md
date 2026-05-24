# `python opt.py models/opt-2.7b c4 --wbits 2` 完整执行流程

## Context
用户希望从头到尾、一步不差地说明运行 `python opt.py models/opt-2.7b c4 --wbits 2` 时，GPTQ 量化算法的全部计算过程。本文档按代码执行顺序，逐行追踪每一步发生了什么。

---

## 第一阶段：命令行参数解析（opt.py:360-439）

```bash
python opt.py models/opt-2.7b c4 --wbits 2
```

解析后得到关键参数：
| 参数 | 值 |
|------|-----|
| `model` | `"models/opt-2.7b"` |
| `dataset` | `"c4"` |
| `wbits` | `2` |
| `nsamples` | `128`（默认） |
| `seed` | `0`（默认） |
| `percdamp` | `0.01`（默认） |
| `groupsize` | `-1`（默认，表示整行一组） |
| `sym` | `False`（默认，非对称量化） |
| `actorder` | `False`（默认） |
| `nearest` | `False`（默认） |
| `trits` | `False`（默认） |
| `save` | `""`（不保存） |
| `load` | `""`（不加载已量化模型） |
| `benchmark` | `0`（不做基准测试） |

---

## 第二阶段：加载模型（opt.py:441-445）

调用 `get_opt("models/opt-2.7b")`：

1. **禁用所有参数初始化函数**（opt.py:16-20）：
   - `torch.nn.init.kaiming_uniform_` → `skip`
   - `torch.nn.init.uniform_` → `skip`
   - `torch.nn.init.normal_` → `skip`
   - 因为加载预训练权重，不需要随机初始化，加速加载

2. **加载模型**（opt.py:22）：
   ```python
   model = OPTForCausalLM.from_pretrained("models/opt-2.7b", torch_dtype='auto', cache_dir=CACHE_DIR)
   ```
   - OPT-2.7B 模型有 32 层 decoder layers
   - `torch_dtype='auto'` → 通常加载为 float16
   - 模型结构：embed_tokens → embed_positions → 32 × OPTDecoderLayer → final_layer_norm → lm_head
   - 每层 decoder layer 包含：self_attn（q_proj, k_proj, v_proj, out_proj）+ FFN（fc1, fc2）

3. **设置序列长度**（opt.py:23）：
   ```python
   model.seqlen = model.config.max_position_embeddings  # = 2048
   ```

4. **设为评估模式**（opt.py:445）：
   ```python
   model.eval()  # 关闭 dropout 等
   ```

---

## 第三阶段：加载校准数据（opt.py:447-449）

调用 `get_loaders("c4", nsamples=128, seed=0, model="models/opt-2.7b", seqlen=2048)`

内部调用 `get_c4(nsamples=128, seed=0, seqlen=2048, model="models/opt-2.7b")`：

### 3.1 加载 C4 数据集（datautils.py:56-115）

1. **尝试加载本地 arrow 文件**：
   ```python
   traindata = Dataset.from_file(".../json-train.arrow")
   valdata = Dataset.from_file(".../json-validation.arrow")
   ```

2. **加载 tokenizer**：
   ```python
   tokenizer = AutoTokenizer.from_pretrained("models/opt-2.7b", use_fast=False)
   ```

3. **采样 128 条校准数据**（datautils.py:82-95）：
   ```python
   random.seed(0)
   trainloader = []
   for _ in range(128):
       while True:
           i = random.randint(0, len(traindata) - 1)         # 随机选一条文本
           trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
           if trainenc.input_ids.shape[1] >= 2048:            # 确保够长
               break
       i = random.randint(0, trainenc.input_ids.shape[1] - 2049)
       j = i + 2048
       inp = trainenc.input_ids[:, i:j]                       # shape: (1, 2048)
       tar = inp.clone()
       tar[:, :-1] = -100                                     # 只有最后一个 token 的 target 有效
       trainloader.append((inp, tar))
   ```
   - 最终得到 `trainloader`：长度为 128 的列表，每个元素是 `(input_ids, target_ids)`，shape 都是 `(1, 2048)`

4. **采样 256 条验证数据**（datautils.py:98-113）：
   ```python
   random.seed(0)
   valenc = []
   for _ in range(256):
       # 同样方式采样，每条长度 2048
       ...
   valenc = torch.hstack(valenc)   # shape: (1, 256×2048) = (1, 524288)
   valenc = TokenizerWrapper(valenc)
   ```

---

## 第四阶段：GPTQ 逐层量化（opt.py:451-454）

```python
if args.wbits < 16 and not args.nearest:  # 2 < 16 且非 nearest → 进入量化
    tick = time.time()
    quantizers = opt_sequential(model, dataloader, DEV)
```

调用 `opt_sequential(model, dataloader, dev='cuda:0')`，这是 GPTQ 的核心入口。

### 4.1 准备阶段：捕获第一层的输入激活（opt.py:28-76）

**目的**：用校准数据前向传播到第一个 decoder layer，记录该层接收到的 128 个输入张量，作为逐层量化的起始点。

1. **关闭 KV cache**（opt.py:30-31）：
   ```python
   use_cache = model.config.use_cache
   model.config.use_cache = False
   layers = model.model.decoder.layers   # 32 层
   ```

2. **将 embedding 层移到 GPU**（opt.py:34-39）：
   ```python
   model.model.decoder.embed_tokens = embed_tokens.to('cuda:0')
   model.model.decoder.embed_positions = embed_positions.to('cuda:0')
   # project_in/project_out 如果存在也移到 GPU
   ```

3. **将第一层 decoder layer 移到 GPU**（opt.py:40）：
   ```python
   layers[0] = layers[0].to('cuda:0')
   ```

4. **准备输入缓冲区**（opt.py:42-46）：
   ```python
   dtype = next(iter(model.parameters())).dtype          # float16
   inps = torch.zeros(
       (128, 2048, 2560), dtype=float16, device='cuda:0'  # 128个样本 × 2048序列长 × 2560隐藏维度
   )
   cache = {'i': 0, 'attention_mask': None}
   ```
   - OPT-2.7B 的 `hidden_size = 2560`

5. **安装 Catcher 钩子**（opt.py:48-57）：
   ```python
   class Catcher(nn.Module):
       def __init__(self, module):
           super().__init__()
           self.module = module
       def forward(self, inp, **kwargs):
           inps[cache['i']] = inp                    # 捕获输入
           cache['i'] += 1
           cache['attention_mask'] = kwargs['attention_mask']
           raise ValueError                           # 中断前向传播！
   layers[0] = Catcher(layers[0])   # 用 Catcher 替换第一层
   ```
   - Catcher 的作用：当模型前向传播到第一层 decoder layer 时，**截获输入并立即抛出异常**，避免后续层计算
   - 这样每个校准样本只需要经过 embedding + 第一层就停止

6. **前向传播 128 个校准样本**（opt.py:58-62）：
   ```python
   for batch in dataloader:   # 128 个 (input_ids, target_ids)
       try:
           model(batch[0].to('cuda:0'))   # 完整模型前向传播
       except ValueError:
           pass   # 捕获 Catcher 抛出的异常
   ```
   每个样本的执行路径：
   - `batch[0]` shape: `(1, 2048)` → token IDs
   - embed_tokens: `(1, 2048)` → `(1, 2048, 2560)` → token embedding
   - embed_positions: 位置编码 → 加到 token embedding
   - 进入 layers[0]（即 Catcher）→ Catcher.forward 接收 `(1, 2048, 2560)` 的输入
   - `inps[i] = inp` → 存储第 i 个样本的第一层输入
   - 抛出 ValueError，被外层 try/except 捕获

7. **清理**（opt.py:63-76）：
   ```python
   layers[0] = layers[0].module   # 恢复原始的第一层
   layers[0] = layers[0].cpu()    # 移回 CPU 释放 GPU 显存
   # embedding 层也移回 CPU
   torch.cuda.empty_cache()
   ```

**结果**：`inps` 是一个 `(128, 2048, 2560)` 的张量，存储了 128 个校准样本在第一层 decoder layer 的输入激活。

### 4.2 逐层量化主循环（opt.py:79-121）

```python
outs = torch.zeros_like(inps)    # (128, 2048, 2560)
attention_mask = cache['attention_mask']
quantizers = {}

for i in range(32):   # 遍历 32 层 decoder layers
```

对每一层 `layers[i]`，执行以下步骤：

#### 步骤 A：移到 GPU 并找出可量化的线性层（opt.py:81-83）

```python
layer = layers[i].to('cuda:0')
subset = find_layers(layer)   # 找出所有 nn.Linear 层
```

`find_layers`（modelutils.py:8-16）递归搜索，返回一个字典。对于 OPT 的每层 decoder layer，会找到：
- `self_attn.q_proj` — shape `(2560, 2560)` 的 Linear
- `self_attn.k_proj` — shape `(2560, 2560)` 的 Linear
- `self_attn.v_proj` — shape `(2560, 2560)` 的 Linear
- `self_attn.out_proj` — shape `(2560, 2560)` 的 Linear
- `fc1` — shape `(10240, 2560)` 的 Linear（FFN 第一层）
- `fc2` — shape `(2560, 10240)` 的 Linear（FFN 第二层）

共 6 个线性层。

#### 步骤 B：为每个线性层创建 GPTQ 对象和 Quantizer（opt.py:84-90）

```python
gptq = {}
for name in subset:   # 6 个线性层
    gptq[name] = GPTQ(subset[name])   # 创建 GPTQ 实例
    gptq[name].quantizer = Quantizer()
    gptq[name].quantizer.configure(
        wbits=2, perchannel=True, sym=False, mse=False, trits=False
    )
```

**GPTQ.__init__**（gptq.py:21-32）对每个线性层做了什么：
```python
W = layer.weight.data.clone()           # 克隆权重矩阵
# 对于 nn.Linear，W 保持原样
self.rows = W.shape[0]                  # 输出维度
self.columns = W.shape[1]               # 输入维度
self.H = torch.zeros((self.columns, self.columns))  # Hessian 矩阵初始化为全零
self.nsamples = 0
```

例如对于 `q_proj`：
- `W` shape: `(2560, 2560)`
- `self.rows = 2560`, `self.columns = 2560`
- `self.H` shape: `(2560, 2560)`，全零

**Quantizer.configure**（quant.py:20-34）做了什么：
```python
self.maxq = torch.tensor(2 ** 2 - 1)   # = 3（2-bit 量化，量化级别 0,1,2,3）
self.perchannel = True                  # 每行独立的 scale 和 zero
self.sym = False                        # 非对称量化
self.mse = False                        # 不用 MSE 搜索最优参数
```

#### 步骤 C：注册前向钩子收集 Hessian 信息（opt.py:92-102）

```python
def add_batch(name):
    def tmp(_, inp, out):
        gptq[name].add_batch(inp[0].data, out.data)
    return tmp

handles = []
for name in subset:
    handles.append(subset[name].register_forward_hook(add_batch(name)))
```

钩子的作用：当线性层 `subset[name]` 执行前向传播时，自动调用 `add_batch(name)`，将 `(输入, 输出)` 传入 `GPTQ.add_batch`。

#### 步骤 D：前向传播 128 个样本收集激活（opt.py:99-102）

```python
for j in range(128):
    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
```

每次前向传播：
- `inps[j].unsqueeze(0)` shape: `(1, 2048, 2560)` → 第 j 个样本
- 经过当前 decoder layer 的所有子模块（self_attn → FFN）
- 中间每经过一个 Linear 层，钩子触发 `add_batch`
- `outs[j]` = 该层的输出，shape: `(1, 2048, 2560)` → `[0]` 取 `(2048, 2560)`

**注意**：`outs[j]` 存储的是整层的输出，而非单个线性层的输出。

##### GPTQ.add_batch 的详细计算（gptq.py:34-60）

对于每个线性层（例如 `q_proj`，权重 shape `(2560, 2560)`）：

```python
def add_batch(self, inp, out):
    # inp: 线性层的输入，shape (1, 2048, 2560)
    # out: 线性层的输出，shape (1, 2048, 2560)（原始 FP16 输出，用于调试）

    if len(inp.shape) == 2:
        inp = inp.unsqueeze(0)
    tmp = inp.shape[0]                    # = 1（batch_size）

    # 对于 nn.Linear：
    if len(inp.shape) == 3:
        inp = inp.reshape((-1, inp.shape[-1]))  # (2048, 2560)
    inp = inp.t()                                # (2560, 2048)

    # 更新 Hessian 的累积
    self.H *= self.nsamples / (self.nsamples + tmp)   # 对已有 H 缩放
    self.nsamples += tmp                               # nsamples += 1（本次的 batch）
    inp = math.sqrt(2 / self.nsamples) * inp.float()  # 缩放因子
    self.H += inp.matmul(inp.t())                      # H += inp @ inp^T
```

**Hessian 累积公式**：
$$H^{(t)} = \frac{t-1}{t} H^{(t-1)} + \frac{2}{t} X_t X_t^T$$

其中 $X_t$ 是第 $t$ 个样本的激活矩阵（shape `2560 × 2048`）。

展开来看，128 个样本全部累积后：
$$H = \frac{2}{128} \sum_{t=1}^{128} X_t X_t^T$$

这就是 **Fisher 信息矩阵的近似**（也是 Hessian 矩阵的近似），因为 GPTQ 最小化的是 $\text{tr}((W_q - W)^T H (W_q - W))$。

#### 步骤 E：移除钩子（opt.py:101-102）

```python
for h in handles:
    h.remove()
```

#### 步骤 F：对每个线性层执行 GPTQ 量化（opt.py:104-111）

```python
for name in subset:
    print(i, name)
    print('Quantizing ...')
    gptq[name].fasterquant(
        percdamp=0.01, groupsize=-1, actorder=False, static_groups=False
    )
    quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
    gptq[name].free()
```

对 6 个线性层依次执行 `fasterquant`。

##### fasterquant 的完整计算过程（gptq.py:62-165）

这是 GPTQ 算法的核心。以 `q_proj`（权重 shape `(2560, 2560)`）为例：

**F1. 准备权重和量化参数**

```python
W = self.layer.weight.data.clone()   # (2560, 2560)
W = W.float()                        # 转为 float32 以保证精度
```

**F2. 初始化全局量化参数**（因为 `ready()` 返回 False，scale 还是全零）

```python
self.quantizer.find_params(W, weight=True)
```

`find_params`（quant.py:36-108）的计算：
```python
# perchannel=True, weight=True → x = W.flatten(1) = W  # (2560, 2560)
x = W   # (2560, 2560)

xmin = x.min(1)[0]    # 每行的最小值，shape (2560,)
xmax = x.max(1)[0]    # 每行的最大值，shape (2560,)

# 非对称量化 (sym=False)：
xmin = minimum(xmin, 0)    # 确保 ≤ 0
xmax = maximum(xmax, 0)    # 确保 ≥ 0

# 处理全零行：
# 如果 xmin==0 且 xmax==0，设 xmin=-1, xmax=1

# maxq = 3 (2-bit)，非 trits 模式：
scale = (xmax - xmin) / 3    # 每行的 scale，shape (2560,)
zero = round(-xmin / scale)  # 每行的 zero point，shape (2560,)

# reshape 为列向量：
scale = scale.reshape(-1, 1)  # (2560, 1)
zero = zero.reshape(-1, 1)    # (2560, 1)
```

**F3. 处理 Hessian 中的死列**

```python
H = self.H                           # (2560, 2560)
dead = torch.diag(H) == 0            # Hessian 对角线为 0 → 该列没有激活
H[dead, dead] = 1                    # 防止除零
W[:, dead] = 0                       # 死列的权重直接置零
```

**F4. Cholesky 分解求 Hessian 逆**

这是 GPTQ 的关键数学步骤：

```python
damp = 0.01 * torch.mean(torch.diag(H))   # 阻尼系数 = 1% × Hessian 对角线均值
H[diag, diag] += damp                      # H ← H + damp × I（正则化）

H = torch.linalg.cholesky(H)              # Cholesky 分解：H = L @ L^T → H 现在是 L
H = torch.cholesky_inverse(H)             # H = L^{-1} @ L^{-T} = (L @ L^T)^{-1} = H^{-1}
H = torch.linalg.cholesky(H, upper=True)  # 对 H^{-1} 做 Cholesky 分解（上三角）
Hinv = H                                   # 最终得到 Hinv，是 H^{-1} 的上三角 Cholesky 分解
```

数学含义：
- $H_{inv} = U^T U$，其中 $U$ 是上三角矩阵
- 这样 $H_{inv}$ 对角线元素 $d_i = U_{ii}$ 就是后续需要的缩放因子

**F5. 分块量化（核心循环）**

```python
blocksize = 128
for i1 in range(0, 2560, 128):      # 列索引：0, 128, 256, ..., 2432
    i2 = min(i1 + 128, 2560)         # 当前块的结束列
    count = i2 - i1                   # 通常是 128
```

每个 block 内部的逐列量化：

```python
W1 = W[:, i1:i2].clone()             # 当前块的权重列，shape (2560, 128)
Q1 = torch.zeros_like(W1)            # 量化后的权重
Err1 = torch.zeros_like(W1)          # 每列的量化误差
Hinv1 = Hinv[i1:i2, i1:i2]          # 当前块的 Hessian 逆子矩阵，shape (128, 128)

for i in range(count):               # 逐列处理 block 内的 128 列
    w = W1[:, i]                     # 当前列的权重向量，shape (2560,)
    d = Hinv1[i, i]                  # Hessian 逆的对角线元素（标量）
```

**对每一列的具体操作**：

**(a) 量化**

```python
# groupsize = -1，使用全局 scale 和 zero
q = quantize(w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq).flatten()
```

`quantize` 函数（quant.py:6-10）：
```python
# maxq = 3 (≥0)，走正常量化路径
q = torch.clamp(torch.round(w / scale) + zero, 0, 3)   # 量化到 0,1,2,3
q = scale * (q - zero)                                   # 反量化回浮点
```

具体计算（对每行 r 独立）：
$$q_r = \text{clamp}\left(\text{round}\left(\frac{w_r}{\text{scale}_r}\right) + \text{zero}_r, 0, 3\right)$$
$$\hat{w}_r = \text{scale}_r \times (q_r - \text{zero}_r)$$

**(b) 计算量化误差**

```python
Q1[:, i] = q                                    # 存储量化结果
Losses1[:, i] = (w - q) ** 2 / d ** 2           # 加权误差
```

$$\text{loss}_i = \frac{(w_i - q_i)^2}{d_i^2}$$

**(c) 误差传播（GPTQ 的核心创新）**

```python
err1 = (w - q) / d                              # 误差向量，shape (2560,)
W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
Err1[:, i] = err1
```

数学含义：
$$W[:, i:] \leftarrow W[:, i:] - \frac{w - q}{d} \cdot H_{inv}[i, i:]$$

这表示：**量化第 i 列产生的误差，通过 Hessian 逆矩阵传播到后续所有未量化的列**。这就是 GPTQ 的 OBQ (Optimal Brain Quantization) 方法——利用 Hessian 信息来最优地补偿量化误差。

**(d) 块间误差传播**

当一个 block 的 128 列全部量化完成后：
```python
Q[:, i1:i2] = Q1
Losses[:, i1:i2] = Losses1 / 2

W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
```

数学含义：
$$W[:, i_2:] \leftarrow W[:, i_2:] - \text{Err1} @ H_{inv}[i_1:i_2, i_2:]$$

这表示将当前块内所有列的量化误差，传播到后续所有未量化的块。

**F6. 写回量化权重**

```python
self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
```

量化后的 2-bit 权重（以 scale * (q - zero) 的浮点形式）写回模型。

**F7. 调试输出**

```python
if DEBUG:
    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))   # 实际 MSE
    print(torch.sum(Losses))                                      # 理论 MSE
```

#### 步骤 G：用量化后的层重新前向传播，更新 outs（opt.py:112-113）

```python
for j in range(128):
    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
```

这一步至关重要：**用量化后的权重重新计算输出**，使得下一层接收到的输入包含了当前层量化的误差影响。

#### 步骤 H：清理并交换 inps/outs（opt.py:115-120）

```python
layers[i] = layer.cpu()      # 移回 CPU
del layer
del gptq
torch.cuda.empty_cache()

inps, outs = outs, inps      # 交换！当前层的输出变成下一层的输入
```

### 4.3 量化循环总结

整个逐层量化过程：

```
layers[0]: inps(原始128样本) → 量化6个Linear → outs(量化后的输出)
           inps, outs = outs, inps

layers[1]: inps(layers[0]的输出) → 量化6个Linear → outs
           inps, outs = outs, inps

...

layers[31]: inps(layers[30]的输出) → 量化6个Linear → outs
```

总共量化：32 层 × 6 个 Linear = **192 个线性层**。

---

## 第五阶段：评估（opt.py:468-476）

```python
datasets = ['c4']
for dataset in datasets:
    dataloader, testloader = get_loaders('c4', seed=0, model="models/opt-2.7b", seqlen=2048)
    opt_eval(model, testloader, DEV)
```

### opt_eval 的计算过程（opt.py:126-231）

**目的**：在测试集上计算 perplexity（困惑度）。

1. **捕获量化后所有层的逐层激活**（与 opt_sequential 类似）：
   - 用 Catcher 捕获第一层的输入
   - 逐层前向传播，每层都将量化后的权重应用

2. **计算 perplexity**（opt.py:211-228）：
   ```python
   nlls = []
   for i in range(nsamples):   # nsamples = 524288 // 2048 = 256
       hidden_states = inps[i].unsqueeze(0)             # (1, 2048, 2560)
       hidden_states = final_layer_norm(hidden_states)   # LayerNorm
       lm_logits = lm_head(hidden_states)               # (1, 2048, 50272) 词表大小

       # 计算 cross-entropy loss（只看下一个 token 预测）
       shift_logits = lm_logits[:, :-1, :]              # (1, 2047, 50272)
       shift_labels = testenc[:, (i*2048):((i+1)*2048)][:, 1:]  # (1, 2047)

       loss = CrossEntropyLoss(shift_logits.view(-1, 50272), shift_labels.view(-1))
       nlls.append(loss.float() * 2048)

   ppl = torch.exp(torch.stack(nlls).sum() / (256 * 2048))
   print(ppl.item())
   ```

   公式：$$\text{PPL} = \exp\left(\frac{\sum_i \text{NLL}_i}{N}\right)$$

---

## 第六阶段：不执行的操作

由于 `args.save = ""` 且 `args.benchmark = 0`：
- 不执行保存（opt.py:478-480）
- 不执行基准测试（opt.py:456-464）

---

## 完整数据流总结

```
输入命令: python opt.py models/opt-2.7b c4 --wbits 2
         │
         ▼
┌─── 加载 OPT-2.7B 模型 (float16, 32层) ───┐
│   embed_tokens → embed_positions           │
│   32 × OPTDecoderLayer                     │
│     └─ self_attn: q,k,v,out_proj (4×Linear)│
│     └─ ffn: fc1, fc2 (2×Linear)            │
│   final_layer_norm → lm_head               │
└────────────────────────────────────────────┘
         │
         ▼
┌─── 加载 C4 校准数据 (128条, 每条2048 tokens) ──┐
│   随机采样 C4 子集 → tokenizer → (1, 2048) 张量│
└────────────────────────────────────────────────┘
         │
         ▼
┌─── 捕获第一层输入激活 ───┐
│   128个样本 × (2048, 2560) │
│   存入 inps 张量            │
└────────────────────────────┘
         │
         ▼
    ┌──── For each of 32 layers ────┐
    │                                │
    │  ① 找出 6 个 Linear 层          │
    │  ② 创建 GPTQ + Quantizer 对象   │
    │     - maxq = 3 (2-bit)          │
    │     - perchannel=True            │
    │     - sym=False                  │
    │                                │
    │  ③ 注册钩子，前向128个样本        │
    │     累积 Hessian:                │
    │     H = (2/N) Σ X_t @ X_t^T    │
    │                                │
    │  ④ 对每个 Linear 层量化:         │
    │     a. find_params → scale,zero  │
    │     b. Cholesky → H^{-1}         │
    │     c. 分块(128列)逐列量化:       │
    │        - quantize(w) → q         │
    │        - err = (w-q)/d           │
    │        - W[:,i:] -= err ⊗ Hinv[i,:] │
    │     d. 写回 Q 到模型权重          │
    │                                │
    │  ⑤ 用量化后权重重算 outs          │
    │  ⑥ inps,outs = outs,inps        │
    │                                │
    └────────────────────────────────┘
         │
         ▼
┌─── 评估 Perplexity ───┐
│   C4 测试集 (256条)     │
│   逐层前向传播           │
│   → lm_head → CE loss  │
│   → PPL = exp(avg NLL) │
└────────────────────────┘
         │
         ▼
      输出 PPL 数值
```

---

## 关键数学公式汇总

### 1. 量化/反量化
$$\hat{w} = \text{scale} \times \left(\text{clamp}\left(\text{round}\left(\frac{w}{\text{scale}}\right) + \text{zero}, 0, 2^b - 1\right) - \text{zero}\right)$$

2-bit 时：$b=2$，量化级别 $0, 1, 2, 3$

### 2. Scale 和 Zero Point 计算
$$\text{scale} = \frac{x_{\max} - x_{\min}}{2^b - 1}$$
$$\text{zero} = \text{round}\left(\frac{-x_{\min}}{\text{scale}}\right)$$

### 3. Hessian 累积
$$H = \frac{2}{N} \sum_{t=1}^{N} X_t X_t^T$$

### 4. 误差传播（OBQ 核心）
$$W_{[:,i:]} \leftarrow W_{[:,i:]} - \frac{w_i - q_i}{H_{inv}[i,i]} \cdot H_{inv}[i, i:]$$

### 5. Perplexity
$$\text{PPL} = \exp\left(\frac{1}{N}\sum_{i} \text{NLL}_i\right)$$
