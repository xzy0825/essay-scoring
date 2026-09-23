# 自动作文评分

根据学生议论文预测 1–6 分的综合分。评价指标是 Quadratic Weighted Kappa（QWK）：差 1 分的惩罚轻，差很多分的惩罚重。1 表示完全一致，0 表示和随机差不多。

这是 AI Coding Gym 上的 MLE-bench 题目 `learning-agency-lab-automated-essay-scoring-2`。本地数据包是比赛划分：训练集 15576 篇，测试集 1731 篇。分数很不均衡，3 分最多，6 分只有 135 篇。

## 结果

折外预测上的 QWK：

- TF-IDF + LightGBM，四舍五入：**0.778**；阈值切分：**0.803**
- DeBERTa-v3-small，四舍五入：**0.797**；阈值切分：**0.809**
- 两者平均，四舍五入：**0.802**；阈值切分：**0.820**

`outputs/submission.csv` 用的是平均模型加阈值切分。阈值是在同一批折外预测上搜出来的，所以 0.820 会比真正没见过的测试集略乐观。四舍五入的 0.802 是更保守的数字。这份文件提交后的线上分数是 **0.832**。

## 复现

环境用 uv，Python 3.11。PyTorch 走 cu121 轮子，对应这台机器的驱动 535 和 RTX 3080。

```bash
uv python install 3.11
uv sync
```

数据：

```bash
uv tool install aicodinggym-cli
aicodinggym configure --user-id YOUR_USER_ID
aicodinggym mle download learning-agency-lab-automated-essay-scoring-2
```

下载接口本身不需要登录。本次数据来自 `https://aicodinggym.com/api/competitions/learning-agency-lab-automated-essay-scoring-2/download`，解压在 `data/`。把分数交回排行榜时才需要 User ID。

训练和交卷：

```bash
uv run python -m src.train_lgbm
HF_ENDPOINT=https://hf-mirror.com uv run python -m src.train_deberta
uv run python -m src.submit
```

这台机器访问 huggingface.co 会被重置连接，所以拉 DeBERTa 权重时加了镜像。镜像可用时不必设置 `HF_ENDPOINT`。

## 为什么这样建模

按 `score` 做分层 5 折，种子 42。1 分和 6 分很少，随机划分会让某一折几乎没有这些样本，QWK 会抖。

模型做回归，不直接做 6 分类。QWK 关心的是分差，回归输出的连续分数保留了“3.6 比 3.1 更接近 4 分”这种信息。交卷前再变成 1–6 的整数。

向量化器只在每一折的训练集上拟合。LightGBM 用词 n-gram、字符 n-gram，再加上篇幅、词数、句数、段落数。DeBERTa-v3-small 取最后一层的 mean pooling，用 MSE 回归，fp16 前向，参数保持 float32。官方权重只有 `pytorch_model.bin`，当前 transformers 在 torch 2.5 上拒绝直接 `torch.load`，脚本会先把它转成 safetensors。

整数分数由连续预测切出来。具体的折划分、序列长度和按题目校准见下一节。上面的 0.778 / 0.797 / 0.820 / 0.832 是 `main` 分支的结果。

## 这一支做了什么

`main` 上的 0.832 把早停、选 epoch 和阈值搜索放在同一份折外结果上。这一支把这几件事拆开，并改了三处模型：

- 每个训练折再留 10% 只做 LightGBM 早停和 DeBERTa 选 epoch。树的棵数或 epoch 数确定后，LightGBM 会在整个训练折上重训。第 0 折不参与混合权重和阈值搜索，只用来报留出 QWK。
- DeBERTa 最大长度改为 1024，batch 为 2，梯度累积为 8。回归头在 mean pooling 后面拼上字符数、词数、句数和段落数。
- 用作文里的题目用语把样本分到 PERSUADE 常见题目上。校准折里样本不少于 200 的题目单独搜切分点，其余用全局切分点。对不上题目的作文保持 `unknown`。

训练结果写在 `outputs/opt/`，不覆盖 `main` 那次的 `outputs/submission.csv`。

这一支的留出折（第 0 折，没有参与选权重和阈值）四舍五入 QWK 是 **0.815**，按题目切分后是 **0.823**。五折整体四舍五入是 DeBERTa **0.812**、LightGBM **0.776**。校准折上 DeBERTa 的权重被选成 1.0，提交文件没有再混入 LightGBM。线上分数是 **0.821**，低于 `main` 的 **0.832**。`main` 仍是分数更高的那一版。1024 长度只碰到 112 篇作文的上限，大部分作文本来就短于 512，所以加长带来的折外提升没有留到线上。

```bash
uv run python -m src.train_lgbm
HF_ENDPOINT=https://hf-mirror.com uv run python -m src.train_deberta
uv run python -m src.submit
```

伪标签还没做。训练集主体是 Persuade，测试集里还有另一批来源，评分尺度可能不同。如果留出折和线上分都没有超过 0.832，再把测试集里预测靠近整数的作文加回去训第二轮。
