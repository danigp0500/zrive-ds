"""
Quick visualization of dataset previous to a
complete analysis in Jupyter Notebook
"""

import pandas as pd
import os

# Data directory
data_dir = os.path.join(os.path.dirname(__file__), "data")

# 1) Load archives
orders = pd.read_parquet(os.path.join(data_dir, "orders.parquet"))
inventory = pd.read_parquet(os.path.join(data_dir, "inventory.parquet"))
users = pd.read_parquet(os.path.join(data_dir, "users.parquet"))
regulars = pd.read_parquet(os.path.join(data_dir, "regulars.parquet"))
abandoned = pd.read_parquet(os.path.join(data_dir, "abandoned_carts.parquet"))

# 2) Quick check (Structure and data)

""" ORDERS """
orders_shape = orders.shape
print(f"\n'Orders' dimensions: {orders_shape}")
orders_columns = orders.columns.to_list()
print(f"\n'Orders' columns: {orders_columns}")
orders_dtypes = orders.dtypes
print(f"\n'Orders' dtypes: {orders_dtypes}")
orders_nan = orders.isna().sum()
print(f"\n'Orders' missing values: {orders_nan}")
orders_head = orders.head()
print(f"\n'Orders' first values: {orders_head}")

# 3) After a quick view or previous prints, we begin to ask some questions:
# Is there any order repeated?:
orders_id_dup = orders["user_id"].duplicated().sum()
print(f"\n 'Orders' of same users duplicated: {orders_id_dup}")

# who are the ones who has ordered the most?
orders_counts = orders["user_id"].value_counts()
print(orders_counts.head())
print(orders_counts.describe())

print("\n=============================================")

""" USERS """
users_shape = users.shape
print(f"\n'Users' dimensions: {users_shape}")
users_columns = users.columns.to_list()
print(f"\n'Users' columns: {users_columns}")
users_dtypes = users.dtypes
print(f"\n'Users' dtypes: {users_dtypes}")
users_nan = users.isna().sum()
print(f"\n'Users' missing values: {users_nan}")
users_head = users.head()
print(f"\n'Users' first values: {users_head}")

print("\n=============================================")

""" REGULARS """
regulars_shape = regulars.shape
print(f"\n'Regulars' dimensions: {regulars_shape}")
regulars_columns = regulars.columns.to_list()
print(f"\n'Regulars' columns: {regulars_columns}")
regulars_dtypes = regulars.dtypes
print(f"\n'Regulars' dtypes: {regulars_dtypes}")
regulars_nan = regulars.isna().sum()
print(f"\n'Regulars' missing values: {regulars_nan}")
regulars_head = regulars.head()
print(f"\n'Regulars' first values: {regulars_head}")

# Is there any product repeated?:
regulars_top = regulars["variant_id"].value_counts().head(10)
print(f"\n'Regulars' Top products: \n{regulars_top}")


print("\n=============================================")

""" ABANDONED CARTS """
abandoned_shape = abandoned.shape
print(f"\n'Abandoned' dimensions: {abandoned_shape}")
abandoned_columns = abandoned.columns.to_list()
print(f"\n'Abandoned' columns: {abandoned_columns}")
abandoned_dtypes = abandoned.dtypes
print(f"\n'Abandoned' dtypes: {abandoned_dtypes}")
abandoned_nan = abandoned.isna().sum()
print(f"\n'Abandoned' missing values: {abandoned_nan}")
abandoned_head = abandoned.head()
print(f"\n'Abandoned' first values: {abandoned_head}")

print("\n=============================================")

""" INVENTORY """
inventory_shape = inventory.shape
print(f"\n'inventory' dimensions: {inventory_shape}")
inventory_columns = inventory.columns.to_list()
print(f"\n'inventory' columns: {inventory_columns}")
inventory_dtypes = inventory.dtypes
print(f"\n'inventory' dtypes: {inventory_dtypes}")
inventory_nan = inventory.isna().sum()
print(f"\n'inventory' missing values: {inventory_nan}")
inventory_head = inventory.head()
print(f"\n'inventory' first values: {inventory_head}")

# Most common products
print(inventory["product_type"].value_counts().head(10))

# Top vendoors
print(inventory["vendor"].value_counts().head(10))
