# NBA 球员位置预测

基于 NBA 球员赛季技术统计，预测数据中标注的五类场上位置：中锋（C）、大前锋（PF）、控球后卫（PG）、得分后卫（SG）和小前锋（SF）。这是一个课程实验项目，包含朴素贝叶斯、K-Means、手写 ID3/C4.5 决策树，以及基于 PyTorch 的深度学习与消融实验。

## 项目内容

| 路径 | 内容 |
| --- | --- |
| `scripts/preprocess_nba.py` | 数据清洗、分层划分、标准化和离散化 |
| `experiments/experiment1_bayes.py` | 高斯朴素贝叶斯分类 |
| `experiments/experiment2_kmeans.py` | K-Means 聚类 |
| `experiments/experiment3_id3.py` | ID3 决策树分类 |
| `experiments/experiment4_c45.py` | C4.5 决策树分类 |
| `experiments/experiment5_dnn.py` | 特征工程、DNN 集成、年代专家和 ExtraTrees 概率融合 |
| `experiments/experiment5_ablation.py` | 深度学习方案的消融实验 |
| `scripts/generate_*_ppt.py` | 根据已生成的结果制作汇报幻灯片（可选） |
| `大数据技术实验_综合实验报告.md` | 原始实验报告及方法细节 |

## 准备环境与数据

建议使用 Python 3.11。安装核心依赖：

```bash
python -m pip install -r requirements.txt
```

幻灯片生成脚本另需：

```bash
python -m pip install -r requirements-ppt.txt
```

把原始数据放在项目根目录，文件名为 `NBA_Season_Stats.csv`。预期字段包括 `Year, Player, Pos, Age, Tm, G, MP, FG, FGA, FG%, 3P, 3PA, 3P%, 2P, 2PA, 2P%, eFG%, FT, FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS`。本仓库未附带原始数据：现有文件的确切出处及再分发条款尚未核实。若使用其他来源的数据，字段、年份和样本范围不同，结果也会变化。

## 运行

在项目根目录依次执行：

```bash
python scripts/preprocess_nba.py
python experiments/experiment1_bayes.py
python experiments/experiment2_kmeans.py
python experiments/experiment3_id3.py
python experiments/experiment4_c45.py
python experiments/experiment5_dnn.py
python experiments/experiment5_ablation.py
```

预处理会写入 `processed/`，各实验会写入 `results/`。两者均是生成目录，默认不提交到 Git。深度学习实验包含多轮模型训练和搜索，所需时间及显存明显高于前四个实验；无 CUDA 时可在 CPU 上运行，但会更慢。消融实验依赖实验五的结果，按上述顺序运行。

## 实验结果与本机复核

原项目在相同数据集上的记录如下。本次使用项目原始数据重新运行了预处理及前四个实验，指标与原记录一致。深度学习模型只完成了 45 维特征构造和网络前向运行检查；其下表指标来自原项目训练记录，未在本次完整重训。消融脚本利用本地已有的缓存结果运行成功，也不等于重新训练各个变体。

| 方法 | Accuracy | Macro F1 | 说明 |
| --- | ---: | ---: | --- |
| Gaussian Naive Bayes | 0.4477 | 0.4027 | 26 个原始数值特征 |
| K-Means | — | — | Silhouette 0.2071，ARI 0.0166 |
| ID3 | 0.4357 | 0.4360 | 26 个离散化特征 |
| C4.5 | 0.4327 | 0.4328 | 26 个离散化特征 |
| DNN + probability fusion | 0.7261 | 0.7258 | 45 个筛选特征 |

前四项是基础方法；最后一项还包含特征工程、集成和概率融合，因此不能把指标差异单独归因于神经网络。有关数据处理、特征和结果分析，请参阅[综合实验报告](大数据技术实验_综合实验报告.md)。

## 复现注意事项

- 训练/测试集按位置分层，以 8:2 划分，随机种子为 42。标准化与离散化只在训练集上拟合。
- 模型不以球员姓名或球队作为输入。`Year` 用于年代相关特征。
- 原始数据中若有同一球员的不同赛季或转队记录，随机逐行划分可能使同一球员出现在训练集和测试集中；这里的成绩代表该划分下的表现，不等于对未见过球员的泛化能力。
- 报告引用的 28 张图表随报告提交。其余生成结果、预测明细、模型权重和汇报幻灯片留在本地。
