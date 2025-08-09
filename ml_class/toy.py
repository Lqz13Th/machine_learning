import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from pytorch_tabnet.tab_model import TabNetRegressor
import wandb
from pytorch_tabnet.callbacks import Callback


# 自定义 wandb callback
class WandbCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs):
        wandb.log({k: float(v) for k, v in logs.items()}, step=epoch)


# 初始化 wandb
wandb.init(project="tabnet-california-housing", name="tabnet_run_with_callback")

# 加载数据
data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 这里把y转成二维
y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

model = TabNetRegressor()

wandb_callback = WandbCallback()

model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=['rmse'],
    max_epochs=50,
    patience=5,
    batch_size=512,
    virtual_batch_size=128,
    callbacks=[wandb_callback],
    drop_last=False,
    num_workers=0
)

wandb.finish()
print("训练完成，数据已上传到 wandb！")
