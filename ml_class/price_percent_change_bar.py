import pandas as pd


def px_pct_bar(
        df: pd.DataFrame,
        threshold: float,
) -> pd.DataFrame:
    last_px = df.iloc[0]['price']
    last_ts = df.iloc[0]['transact_time']

    bars = []
    ts_duration = 0
    sum_buy_size = 0
    sum_sell_size = 0

    print(last_px)
    for i in range(len(df)):
        px = df.iloc[i]['price']
        sz = df.iloc[i]['quantity']
        ts = df.iloc[i]['transact_time']
        side = -1 if df.iloc[i]['is_buyer_maker'] else 1  # 判断买卖方向 (True 为卖方主导，False 为买方主导)

        px_pct = (px - last_px) / last_px

        if side == 1:
            sum_buy_size += sz

        else:
            sum_sell_size += sz

        if abs(px_pct) > threshold:
            ts_duration = ts - last_ts

            bar = {
                'price': px,
                'sum_buy_size': sum_buy_size,
                'sum_sell_size': sum_sell_size,
                'timestamp_duration': ts_duration,
                'price_pct_change': px_pct,
                'change_side': 'Buy' if px_pct > 0 else 'Sell'
            }
            bars.append(bar)

            last_px = px
            last_ts = ts
            sum_buy_size = 0
            sum_sell_size = 0

    bars_df = pd.DataFrame(bars)
    return bars_df


if __name__ == "__main__":
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    agg_trade_data = pd.read_csv("C:/Work Files/data/backtest/aggtrade/FILUSDT/FILUSDT-aggTrades-2024-04.csv")

    px_pct_bar(
        df=agg_trade_data,
    )
    print(agg_trade_data)
