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
    df.created_at = df.created_at.astype(str)
    numeric_dates_filter = df.created_at.str.find(" ") == -1
    df.loc[numeric_dates_filter, "created_at"] = pd.to_datetime(
        df[numeric_dates_filter].created_at.astype(float), unit="D", origin="1899-12-30"
    )
    df.created_at = pd.to_datetime(df.created_at)

    # add useful columns
    df["total_discount"] = df["discount"] + df["voucher_discount"]
    df["value"] = df["items"] * df["price"]
    df["discount_rate"] = df["total_discount"] / df["value"]
    df["discounted_value"] = df["value"] - df["total_discount"]
    df["shipping_fee_discount"] = df["total_shipping_fee"] - df["final_shipping_fee"]

    print(df.shape)
    df.to_csv("./data/item_data.csv")
    return df


def aggregate_items(df: pd.DataFrame):
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


def main() -> None:
    item_df = data_preprocessing()
    order_df = aggregate_items(item_df)


if __name__ == "__main__":
    main()
