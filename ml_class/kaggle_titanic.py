import polars as pl
train_df = pl.read_csv("titanic/train.csv")
print(train_df)

df_train = train_df[train_df['target_col'].notnull()]
df_pred = train_df[train_df['target_col'].isnull()]

print(df_train)
print(df_pred)