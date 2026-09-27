# california-house-prices-pytorch
使用 PyTorch 预测加州房价的学习项目
# 加州房价预测（PyTorch）

使用 PyTorch 完成 Kaggle California House Prices 房价预测的学习项目。

通过这个项目练习数据清洗、特征预处理、神经网络训练和 Kaggle 提交文件生成。

## 项目方法

- 使用 17 个数值特征和房屋类型 `Type`。
- 使用 scikit-learn 填充缺失值、标准化数值、对房屋类型进行独热编码。
- 对部分面积和金额特征使用 `log1p` 变换。
- 使用一个包含 64 个隐藏单元的神经网络预测 `log(成交价)`。
- 使用 MSE 损失和 Adam 优化器训练，再用 `exp` 还原价格。

当前版本使用全部训练数据训练一个模型，不包含五折集成或本地验证评分。

## 数据来源

数据来自 [Kaggle California House Prices](https://www.kaggle.com/competitions/california-house-prices/data)。

数据文件未包含在仓库中。请从比赛页面下载并解压，将以下三个文件放入项目的 `data` 文件夹：

```text
california-house-prices-pytorch/
├── house_prices_complete.py
├── requirements.txt
├── README.md
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

训练集包含 47,439 条记录，测试集包含 31,626 条记录，预测目标是 `Sold Price`。

## 运行方法

本项目已在 Python 3.11 环境中运行。

### 1. 获取项目

```bash
git clone https://github.com/mzc668/california-house-prices-pytorch.git
cd california-house-prices-pytorch
```

### 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 3. 准备数据并运行

按照上面的目录结构放好数据，然后运行：

```bash
python house_prices_complete.py
```

程序完成后会在项目根目录生成 `submission.csv`，包含 `Id` 和 `Sold Price` 两列，可上传到 Kaggle。

## 训练参数

| 参数 | 当前设置 |
| --- | --- |
| 隐藏层单元数 | 64 |
| 激活函数 | ReLU |
| 训练轮数 | 20 |
| 批次大小 | 256 |
| 学习率 | 0.001 |
| 优化器 | Adam |

## 实验记录

一次单模型提交的 Kaggle 结果：

| 公开评分 | 私人评分 |
| --- | --- |
| 0.17348 | 0.15922 |

以上为历史提交记录。当前脚本只训练模型并生成预测文件，评分需要在 Kaggle 提交后查看。
