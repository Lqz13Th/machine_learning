import pandas as pd
from tqdm import tqdm


def generate_px_pct_bar(
        df: pd.DataFrame,
        threshold: float,
        window: int,
) -> pd.DataFrame:
    last_px = df.iloc[0]["price"]
    last_ts = df.iloc[0]["transact_time"]

    bars = []
    sum_buy_size = 0
    sum_sell_size = 0

    print(last_px)
    for i in tqdm(range(len(df)), desc='Processing bars'):
        px = df.iloc[i]["price"]
        sz = df.iloc[i]["quantity"]
        ts = df.iloc[i]["transact_time"]
        side = -1 if df.iloc[i]["is_buyer_maker"] else 1  # 判断买卖方向 (True 为卖方主导，False 为买方主导)

        px_pct = (px - last_px) / last_px

        if side == 1:
            sum_buy_size += sz

        else:
            sum_sell_size += sz

        if abs(px_pct) > threshold:
            ts_duration = ts - last_ts

            bar = {
                "price": px,
                "sum_buy_size": sum_buy_size,
                "sum_sell_size": sum_sell_size,
                "timestamp_duration": ts_duration,
                "price_pct_change": px_pct,
                'buy_sell_imbalance': sum_buy_size - sum_sell_size,
                "change_side": 1 if px_pct > 0 else -1,
            }
            bars.append(bar)

            last_px = px
            last_ts = ts
            sum_buy_size = 0
            sum_sell_size = 0

    bars_df = pd.DataFrame(bars)
    bars_df['future_price_pct_change'] = bars_df['price'].shift(-window) / bars_df['price'] - 1
    bars_df = bars_df.dropna()

    return bars_df


if __name__ == "__main__":
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    agg_trade_data = pd.read_csv("C:/Work Files/data/backtest/aggtrade/FILUSDT/FILUSDT-aggTrades-2024-04.csv")
    print(agg_trade_data)

    px_pct_bar = generate_px_pct_bar(
        df=agg_trade_data,
        threshold=0.01,
        window=3,
    )

    print(px_pct_bar)

    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split

    X = px_pct_bar[[
        'price',
        'sum_buy_size',
        'sum_sell_size',
        'timestamp_duration',
        'timestamp_duration',
        'price_pct_change',
        'buy_sell_imbalance',
        'change_side'
    ]]

    y = px_pct_bar['future_price_pct_change']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)

    print("Lasso Coefficients:", lasso.coef_)
    print("Intercept:", lasso.intercept_)

    # 测试集上进行预测
    y_predict = lasso.predict(X_test)

