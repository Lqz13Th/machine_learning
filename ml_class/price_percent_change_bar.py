import pandas as pd


def px_pct_bar(
    df: pd.DataFrame,
):
    last_price = df.iloc[1, 1]
    print(last_price)
    for i in range(len(df.iloc[:, 0].tolist())):
        pass


if __name__ == "__main__":
    pd.set_option("display.max_rows", 5000)
    pd.set_option("expand_frame_repr", False)

    agg_trade_data = pd.read_csv("C:/Work Files/data/backtest/aggtrade/FILUSDT/FILUSDT-aggTrades-2024-04.csv")

    px_pct_bar(
        df=agg_trade_data,
    )
    print(agg_trade_data)
