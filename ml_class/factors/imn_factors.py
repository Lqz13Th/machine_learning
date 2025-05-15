import numpy as np
import pandas as pd


def cal_imn_usdt(lev):
    return 200 * lev

def impact_price_pct_ask(df: pd.DataFrame, imn: float, levels: int = 2) -> pd.Series:
    result = []

    for _, row in df.iterrows():
        cum_value = 0
        impact_price = row[f"lob_asks[0].price"]

        for i in range(levels):
            price = row[f"lob_asks[{i}].price"]
            amount = row[f"lob_asks[{i}].amount"]
            cum_value += price * amount
            impact_price = price
            if cum_value >= imn:
                break

        best_ask = row["lob_asks[0].price"]
        pct = (impact_price - best_ask) / best_ask if best_ask > 0 else float("nan")
        result.append(pct)

    return pd.Series(result, name=f"impact_price_pct_ask@{imn}")


def impact_price_pct_bid(df: pd.DataFrame, imn: float, levels: int = 2) -> pd.Series:
    result = []

    for _, row in df.iterrows():
        cum_value = 0
        impact_price = row[f"lob_bids[0].price"]

        for i in range(levels):
            price = row[f"lob_bids[{i}].price"]
            amount = row[f"lob_bids[{i}].amount"]
            cum_value += price * amount
            impact_price = price
            print(price, amount, cum_value, impact_price)
            if cum_value >= imn:
                break

        best_bid = row["lob_bids[0].price"]
        pct = (best_bid - impact_price) / best_bid if best_bid > 0 else float("nan")
        result.append(pct)

    return pd.Series(result, name=f"impact_price_pct_bid@{imn}")

if __name__ == "__main__":
    df = pd.DataFrame({
        "lob_asks[0].price": [100, 101],
        "lob_asks[0].amount": [3, 5],
        "lob_asks[1].price": [102, 103],
        "lob_asks[1].amount": [5, 10],
        "lob_bids[0].price": [99, 100],
        "lob_bids[0].amount": [2, 4],
        "lob_bids[1].price": [98, 99],
        "lob_bids[1].amount": [10, 15],
    })

    df["impact_price_pct_bid@500"] = impact_price_pct_bid(df, imn=500)
    print(df[["impact_price_pct_bid@500"]])

