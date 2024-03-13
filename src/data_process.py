import pandas as pd


def data_preprocessing() -> pd.DataFrame:
    # read and clean data
    df = pd.read_csv("./data/raw_data.csv")
    print(df.shape)
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df[["order_number", "user_id", "created_at"]] = df[
        ["order_number", "user_id", "created_at"]
    ].astype(str)

    # fix excel dates
    df["created_at"] = df["created_at"].astype(str)
    numeric_dates_filter = df["created_at"].str.find(" ") == -1
    df.loc[numeric_dates_filter, "created_at"] = pd.to_datetime(
        df[numeric_dates_filter]["created_at"].astype(float),
        unit="D",
        origin="1899-12-30",
    )
    df["created_at"] = df["created_at"].astype("datetime64[s]")

    # add useful columns
    df["total_discount"] = df["discount"] + df["voucher_discount"]
    df["value"] = df["items"] * df["price"]
    df["discount_rate"] = df["total_discount"] / df["value"]
    df["discounted_value"] = df["value"] - df["total_discount"]
    df["shipping_fee_discount"] = df["total_shipping_fee"] - df["final_shipping_fee"]

    print(df.shape)
    df.to_csv("./data/item_data.csv")
    return df


def aggregate_orders(df: pd.DataFrame):
    df = df.groupby(
        by=[
            "order_number",
            "created_at",
            "user_id",
            "city",
            "total_shipping_fee",
            "final_shipping_fee",
            "shipping_fee_discount",
        ]
    ).agg(
        all_items=("items", "sum"),
        distinct_items=("items", "size"),
        distinct_categories=("main_category", "nunique"),
        discount=("discount", "sum"),
        voucher_discount=("voucher_discount", "sum"),
        total_discount=("total_discount", "sum"),
        value=("value", "sum"),
        discounted_value=("discounted_value", "sum"),
    )

    df = df.reset_index().set_index("order_number")
    df["final_discount"] = df["total_discount"] + df["shipping_fee_discount"]
    df["paid_value"] = df["discounted_value"] + df["final_shipping_fee"]
    df["discount_rate"] = df["total_discount"] / df["value"]
    df["final_discount_rate"] = df["final_discount"] / df["paid_value"]

    print(df.shape)
    df.to_csv("./data/order_data.csv")
    return df


def aggregate_customers(df: pd.DataFrame) -> pd.DataFrame:
    # calculate order interval for each customer
    df.sort_values(by=["user_id", "created_at"], inplace=True)
    df["order_interval"] = df.groupby("user_id")["created_at"].diff()

    df = df.groupby(by="user_id").agg(
        first_order=("created_at", "min"),
        last_order=("created_at", "max"),
        orders_count=("created_at", "count"),
        all_items=("all_items", "sum"),
        avg_items=("distinct_items", "mean"),
        avg_paid_value=("paid_value", "mean"),
        total_paid_value=("paid_value", "sum"),
        avg_shipping_fee=("final_shipping_fee", "mean"),
        avg_discount_rate=("discount_rate", "mean"),
        avg_final_discount=("final_discount", "mean"),
        avg_order_interval=("order_interval", "mean"),
    )

    df["avg_order_interval"] = df["avg_order_interval"].dt.days
    df["days_from_last_order"] = (df["last_order"].max() - df["last_order"]).dt.days

    # define loyal customers
    df["loyal_customer"] = (
        (df["orders_count"] > 1)
        & (df["days_from_last_order"] <= 30)
        & (df["avg_order_interval"] <= 20)
    )
    print(
        f"Loyal customers average orders count: {df[df['loyal_customer']]['orders_count'].mean()}"
    )

    # define churned customers
    df["churned_customer"] = (
        (df["days_from_last_order"] > 2 * df["avg_order_interval"])
        & (df["avg_order_interval"] > 10)
        & (~df["loyal_customer"])
    )
    print(
        f"Churned customers average days from last order: {df[df['churned_customer']]['days_from_last_order'].mean()}"
    )

    # define other customer groups
    df["high_value_customer"] = df["total_paid_value"] > df[
        "total_paid_value"
    ].quantile(0.8)
    df["high_paid_customer"] = df["avg_paid_value"] > df["avg_paid_value"].quantile(0.8)
    df["frequent_order_customer"] = (
        df["avg_order_interval"] < df["avg_order_interval"].quantile(0.3)
    ) & (df["orders_count"] > df["orders_count"].quantile(0.7))
    df["multiple_item_customer"] = df["avg_items"] > df["avg_items"].quantile(0.8)
    df["discount_rate_customer"] = df["avg_discount_rate"] > df[
        "avg_discount_rate"
    ].quantile(0.8)
    df["zero_shipping_fee_customer"] = df["avg_shipping_fee"] == 0

    print(df.shape)
    df.to_csv("./data/customer_data.csv")
    return df


def main() -> None:
    item_df = data_preprocessing()
    order_df = aggregate_orders(item_df)
    customers_df = aggregate_customers(order_df)


if __name__ == "__main__":
    main()
