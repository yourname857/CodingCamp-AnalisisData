import pandas as pd


def create_daily_orders_df(df):
    df['date'] = pd.to_datetime(df['dteday'])
    daily_orders_df = df.groupby('date').agg({'cnt': 'sum'}).reset_index()
    daily_orders_df.rename(columns={'cnt': 'order_count'}, inplace=True)
    return daily_orders_df

def create_sum_order_items_df(df):
    return df.groupby("product_name").agg({"cnt": "sum"}).reset_index()

def create_bygender_df(df):
    return df.groupby("gender").agg({"cnt": "sum"}).reset_index()

def create_rfm_df(df):
    rfm_df = df.groupby("customer_id").agg({"dteday": "max", "cnt": "sum"}).reset_index()
    rfm_df.rename(columns={"dteday": "last_purchase", "cnt": "total_orders"}, inplace=True)
    return rfm_df