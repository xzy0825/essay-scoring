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

向量化器只在每一折的训练集上拟合。LightGBM 用词 n-gram、字符 n-gram，再加上篇幅、词数、句数、段落数。早停看的是这一折的验证集，折外分会略偏高。

DeBERTa-v3-small 取最后一层的 mean pooling，用 MSE 回归。最大长度 512，fp16 前向，参数保持 float32，batch 4，梯度累积 4，学习率 2e-5，每个折训 3 个 epoch，留下验证 QWK 最高的那个 epoch。官方权重只有 `pytorch_model.bin`，而当前 transformers 在 torch 2.5 上拒绝 `torch.load`，脚本会先把它转成 safetensors。

整数化有两种：四舍五入，或者在折外预测上搜索 5 个递增阈值。最终提交比较了 LightGBM、DeBERTa 和二者平均，取得分最高的一种。

## 还可以做的优化

当前方案停在这里，是为了把训练、验证和提交收成一条能看懂的路径。下面几件事有机会超过线上的 0.832，但每一件都会多出一条分支，所以没有写进现有脚本。

验证先拆开。每个训练折再留 10% 只做早停和选 epoch；五折里再留一折只搜阈值和两个模型的混合权重。对外报分的那一折不参与这些选择。现在的 0.820 把选择和评估放在了同一份折外结果上。

模型上优先做三件：

- 最大长度从 512 提到 1024。token 中位数已经到 405，最大值顶在 512，被截掉的多半是写得长的高分作文。`deberta-v3-small` 在 10GB 显存上可以把 batch 降到 2，并用梯度累积补回来。
- 把词数、段落数、句子数拼到 mean pooling 后面。长度和分数绑得很紧，截断之后模型不容易再从正文里看到全文有多长。
- 按题目分开校准。不同命题的给分尺度不一样，一组全局阈值会把某些题整体打高或打低。训练表里没有题目编号，可以用公开的 PERSUADE 语料按原文对上题目，再为每个题目单独搜切分点。

如果这三件仍然过不了 0.832，再考虑两阶段伪标签。训练集主体是 Persuade 语料，测试集里还有另一批来源的作文，评分尺度不完全一样。第一阶段只用训练数据；第二阶段把测试集里预测落在整数附近的作文加回去再训一轮。伪标签的变化只会出现在线上分数里，本地折外分看不到，所以要和留出折的 QWK 分开记录。

多个大模型再做集成的收益通常更小，也更难对照每一次改动。上面几件做完仍不够，再考虑增加一个不同种子或换 `deberta-v3-base`。
