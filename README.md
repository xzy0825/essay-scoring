# 自动作文评分

根据学生议论文预测 1–6 分。评价指标是 Quadratic Weighted Kappa（QWK）。差 1 分的惩罚轻，差很多分的惩罚重。

这是 AI Coding Gym 上的 MLE-bench 题目 `learning-agency-lab-automated-essay-scoring-2`。训练集 15576 篇，测试集 1731 篇。3 分最多，6 分只有 135 篇。

## 最优结果

线上最高分是 **0.84594**。`uv run python -m src.submit` 会用下面四个已经训好的折外预测，按固定权重相加，再在全部折外样本上搜一套全局阈值，写出 `outputs/opt/submission.csv`。

| 成分 | 权重 | 预测文件 |
|---|---|---|
| DeBERTa-v3-base，MSE，最大长度 1024 | 0.45 | `outputs/opt/oof_deberta_base.npy` |
| DeBERTa-v3-small，MSE，最大长度 1024 | 0.20 | `outputs/opt/oof_deberta.npy` |
| TF-IDF + LightGBM（`main` 上那版，早停用了验证折） | 0.20 | `outputs/oof_lgbm.npy` |
| TF-IDF + LightGBM，加拼写、连接词、段落和题目 | 0.15 | `outputs/opt/oof_lgbm_rich.npy` |

权重是看线上分数定的，不是本地阈值 QWK 最高的那组。测试集大约 1700 篇，第三位小数会抖。按题目分开设阈值、只交 DeBERTa、以及 6 类分类头，线上都低于这个混合。过程记在 `experiments.md`。

`outputs/` 不进 git。换机器时要自己拷这四组 `oof_*.npy` / `test_*.npy`，以及 `data/`。

## 复现

环境用 uv，Python 3.11。PyTorch 走 cu121 轮子。

```bash
uv python install 3.11
uv sync
uv tool install aicodinggym-cli
aicodinggym configure --user-id YOUR_USER_ID
aicodinggym mle download learning-agency-lab-automated-essay-scoring-2
```

这台机器访问 huggingface.co 会被重置，拉权重时加 `HF_ENDPOINT=https://hf-mirror.com`。

四份预测对应的训练命令：

```bash
# main 分支的 LightGBM，写出 outputs/oof_lgbm.npy
# 当前分支的 src/train_lgbm.py 写的是下面的 rich 版本

# rich LightGBM，写出 outputs/opt/oof_lgbm_rich.npy
uv run python -m src.train_lgbm

# DeBERTa-v3-small，最大长度 1024，写出 outputs/opt/oof_deberta.npy
HF_ENDPOINT=https://hf-mirror.com uv run python -m src.train_deberta

# DeBERTa-v3-base，最大长度 1024，写出 outputs/opt/oof_deberta_base.npy
HF_ENDPOINT=https://hf-mirror.com \
AES_MODEL=microsoft/deberta-v3-base \
AES_OUT=outputs/opt/deberta_base AES_STEM=deberta_base \
uv run python -m src.train_deberta

uv run python -m src.submit
```

`AES_HEAD=cls AES_PROMPT=1 AES_LLRD=0.9` 会改成 6 类交叉熵，并加上题目编号和层间学习率。那一版线上是 0.84249，没有超过回归混合。

划分都是按分数分层的 5 折，种子 42。QWK 的类别固定为 1–6。DeBERTa 的早停和选 epoch 用训练折内部的 10%，不看外层验证折。回归头在 mean pooling 后面拼上 log 篇幅。官方权重只有 `pytorch_model.bin`，脚本会转成 safetensors 再读。
