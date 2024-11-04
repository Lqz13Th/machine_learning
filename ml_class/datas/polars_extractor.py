import polars as pl


class ParseHFTData:
    def __init__(self):
        pass

    @staticmethod
    def parse_agg_trade_data_binance(file_path: str,) -> pl.DataFrame:
        df_agg_trade = pl.read_csv(file_path)
        selected_columns = df_agg_trade[['price', 'quantity', 'transact_time', 'is_buyer_maker']]
        return selected_columns

    @staticmethod
    def parse_trade_data_tardis(file_path: str, ) -> pl.DataFrame:
        df_trade = pl.read_csv(file_path)
        selected_columns = df_trade[['price', 'amount', 'timestamp', 'side']]
        return selected_columns


if __name__ == '__main__':
    psd = ParseHFTData()
    df = psd.parse_trade_data_tardis(
        "C:/Users/trade/PycharmProjects/DataGrabber/datasets/binance-futures_trades_2024-08-05_FILUSDT.csv.gz"
    )

    print(df)

    data = {
        "product": ["A", "A", "B", "B", "A", "B"],
        "region": ["North", "South", "North", "South", "North", "South"],
        "sales": [100, 200, 150, 250, 300, 100],
    }

    df = pl.DataFrame(data)
    columns = df.columns
    rolling_means = df.with_columns(
        [
            pl.col(column).rolling_mean(2).alias(f"{column}_rolling_mean")
            for column in columns
        ] + [
            pl.col(column).rolling_std(2).alias(f"{column}_rolling_std")
            for column in columns
        ]
    )

    print(rolling_means)


    # result = df.group_by(["product", "region"]).agg(pl.sum("sales").alias("total_sales"))

    # print(result)
    import polars_talib




