"""房价预测单模型版：全部训练数据训练一次，预测测试集，生成 submission.csv。"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 0.001


def train_model(features, labels):
    """学习 log(房价)：保留最基本的 PyTorch 训练循环。"""
    torch.manual_seed(42)
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
    nn.init.zeros_(net[-1].weight)
    nn.init.constant_(net[-1].bias, float(y.mean()))
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        net.train()
        for batch_X, batch_y in loader:



            optimizer.zero_grad()
            loss = loss_fn(net(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    return net


@torch.no_grad()
def predict_log(net, features):
    net.eval()
    X = torch.tensor(features, dtype=torch.float32)
    return net(X).numpy().ravel().astype(np.float64).clip(min=0)


def main():
    torch.set_num_threads(4)
    # 1. 读取数据，选择数值特征和房屋类型。
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    submission = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    numeric_cols = train.select_dtypes(include="number").columns.drop(
        ["Id", "Sold Price", "Zip"]
    ).tolist()
    X = train[numeric_cols + ["Type"]].copy()
    X_test = test[numeric_cols + ["Type"]].copy()
    y = np.log(train["Sold Price"].to_numpy())

    # 2. 修正不合理的负数；跨度较大的面积和金额先取 log1p。
    log_cols = ["Lot", "Total interior livable area", "Tax assessed value",
                "Annual tax amount", "Listed Price", "Last Sold Price"]
    for frame in (X, X_test):
        values = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
        frame[numeric_cols] = values.mask(values < 0)
        frame["Year built"] = frame["Year built"].replace(0, np.nan)
        frame[log_cols] = np.log1p(frame[log_cols])

    # 3. 用库完成缺失值填充、标准化、独热编码。
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler()), numeric_cols),
        ("category", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="Missing"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), ["Type"]),
    ])

    # 4. 用全部训练数据训练一个模型，测试集只使用已拟合的预处理器。
    X_all = preprocess.fit_transform(X)
    test_features = preprocess.transform(X_test)
    net = train_model(X_all, y)
    print(y)

    # 5. 预测测试集：将模型输出的 log(房价) 用 exp 还原成实际房价。
    prices = np.exp(predict_log(net, test_features))
    submission["Sold Price"] = submission["Id"].map(pd.Series(prices, index=test["Id"]))
    assert np.isfinite(submission["Sold Price"]).all(), "提交结果存在缺失或无穷大"
    assert (submission["Sold Price"] > 0).all(), "预测价格必须为正数"
    submission.to_csv(ROOT / "submission.csv", index=False)
    print(f"已生成 {len(submission)} 条预测：{ROOT / 'submission.csv'}", flush=True)


if __name__ == "__main__":
    main()
