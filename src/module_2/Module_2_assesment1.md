# Exploratory Data Analysis - ZriveDS Module 2 (Assesment 1)

This notebook contains an initial exploratory analysis of groceries'datasets gathered: 
`orders`, `users`, `inventory`, `regulars`, `abandoned_carts`.

The goal is to understand user behaviour, detect possible problems inside of datasets and formule hypotesis that can be usefull for further analysis.


## 1) Load data


```python
import pandas as pd
import os

data_dir = os.path.join(os.getcwd(), "data")

orders = pd.read_parquet(os.path.join(data_dir, "orders.parquet"))
users = pd.read_parquet(os.path.join(data_dir, "users.parquet"))
inventory = pd.read_parquet(os.path.join(data_dir, "inventory.parquet"))
regulars = pd.read_parquet(os.path.join(data_dir, "regulars.parquet"))                   
abandoned = pd.read_parquet(os.path.join(data_dir, "abandoned_carts.parquet"))

```

## 2) Dataset overview

We begin with a structural check of each dataset: dimensions, column names, data types and missing values



```python
def dataset_overview(df, name):
    print(f"\n--- {name.upper()} ---")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Dtypes:\n", df.dtypes)
    print("Missing values:\n", df.isna().sum())

dataset_overview(orders, "orders")
dataset_overview(users, "users")
dataset_overview(inventory, "inventory")
dataset_overview(regulars, "regulars")
dataset_overview(abandoned, "abandoned_carts")

```

    
    --- ORDERS ---
    Shape: (8773, 6)
    Columns: ['id', 'user_id', 'created_at', 'order_date', 'user_order_seq', 'ordered_items']
    Dtypes:
     id                         int64
    user_id                   object
    created_at        datetime64[us]
    order_date        datetime64[us]
    user_order_seq             int64
    ordered_items             object
    dtype: object
    Missing values:
     id                0
    user_id           0
    created_at        0
    order_date        0
    user_order_seq    0
    ordered_items     0
    dtype: int64
    
    --- USERS ---
    Shape: (4983, 10)
    Columns: ['user_id', 'user_segment', 'user_nuts1', 'first_ordered_at', 'customer_cohort_month', 'count_people', 'count_adults', 'count_children', 'count_babies', 'count_pets']
    Dtypes:
     user_id                   object
    user_segment              object
    user_nuts1                object
    first_ordered_at          object
    customer_cohort_month     object
    count_people             float64
    count_adults             float64
    count_children           float64
    count_babies             float64
    count_pets               float64
    dtype: object
    Missing values:
     user_id                     0
    user_segment                0
    user_nuts1                 51
    first_ordered_at            0
    customer_cohort_month       0
    count_people             4658
    count_adults             4658
    count_children           4658
    count_babies             4658
    count_pets               4658
    dtype: int64
    
    --- INVENTORY ---
    Shape: (1733, 6)
    Columns: ['variant_id', 'price', 'compare_at_price', 'vendor', 'product_type', 'tags']
    Dtypes:
     variant_id            int64
    price               float64
    compare_at_price    float64
    vendor               object
    product_type         object
    tags                 object
    dtype: object
    Missing values:
     variant_id          0
    price               0
    compare_at_price    0
    vendor              0
    product_type        0
    tags                0
    dtype: int64
    
    --- REGULARS ---
    Shape: (18105, 3)
    Columns: ['user_id', 'variant_id', 'created_at']
    Dtypes:
     user_id               object
    variant_id             int64
    created_at    datetime64[us]
    dtype: object
    Missing values:
     user_id       0
    variant_id    0
    created_at    0
    dtype: int64
    
    --- ABANDONED_CARTS ---
    Shape: (5457, 4)
    Columns: ['id', 'user_id', 'created_at', 'variant_id']
    Dtypes:
     id                     int64
    user_id               object
    created_at    datetime64[us]
    variant_id            object
    dtype: object
    Missing values:
     id            0
    user_id       0
    created_at    0
    variant_id    0
    dtype: int64


## 3) Inicial observations and Questions
- Are there repeated users or orders?
- Do some users order more frequently than others?
- Are some products more common than others?
- How many items do users usually buy?
- Why are there so many missing values in colums like "user_nuts1" and "count_people"?

## 4) Hypotheses and Exploration


### ORDERS dataset:
#### *Hypothesis 1*: A small group of users places the majority of the orders

We will analyze hoy many orders each user has placed to detect a possible power-user pattern.



```python
import matplotlib.pyplot as plt

orders_per_user = orders["user_id"].value_counts()

plt.figure(figsize =(10,5))
plt.scatter(range(len(orders_per_user)),orders_per_user, alpha=0.6)
plt.title("Orders per user")
plt.xlabel("Users")
plt.ylabel("Nº orders")
plt.grid(True)
plt.show()

```


    
![png](Module_2_assesment1_files/Module_2_assesment1_7_0.png)
    


This scatter graph shows the number of orders per user. Each dot represents an user, and its position on the Y-axis indicates how many orders they have placed.

- The majority of users have placed only one or two orders.
- There's a noticeable long tail of users who have placed significantly more orders (up to 25). 
- This pattern suggests the existence of a small group of *loyal* or *repeat* customers, while the majority are likely one-time or casual buyers.

This insight could be relevant for future segmentation strategies or retention-focused marketing initiatives.

### USERS dataset:
#### *Hypothesis 2:* Different types of users (user segments) can have different purchasing behaviors. Identifying which segments predominate can help guide next campaigns or analyses.




```python
segment_counts = users["user_segment"].value_counts()

# Bar graph
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    segment_counts.index,     #['Top Up', 'Proposition']
    segment_counts.values,
    color="steelblue",
    edgecolor="black",
    alpha=0.8
)

# Add values above the bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)


ax.set_title("User segments distribution", fontsize=12)
ax.set_ylabel("Nº users")
ax.set_xlabel("Segment")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()
```


    
![png](Module_2_assesment1_files/Module_2_assesment1_11_0.png)
    


Users are grouped into two segments: *Top Up* and *Proposition*.

- Top Up: Frequent shoppers who regularly purchase to restock essential products.
- Proposition: Less frequent shoppers, often drawn in by specific offers or campaigns.

The chart shows that Top Up users slightly outnumber Proposition users, but it is not something really significant to make a conclusion

#### *Hypothesis 3:* Demographic information was likely not mandatory, making segmentation by household size unreliable unless imputed.


```python
cols = ["count_people","count_adults", "count_children", "count_babies", "count_pets"]
missing = users[cols].isna().mean() * 100
missing.plot(kind="barh", title="% of missing values in demographic fields", figsize=(8, 4), xlim=(0, 100))
```




    <Axes: title={'center': '% of missing values in demographic fields'}>




    
![png](Module_2_assesment1_files/Module_2_assesment1_14_1.png)
    


This demographic information was likely optional and rarely filled in, so it cannot be used reliably unless cleaned or imputed.

### INVENTORY dataset:
#### *Hypothesis 4:* If product *price* is lower than *compare_at_price*, that means the product is on sale.



```python
# New boolean column to indicate wether the product is on sale or not
inventory["is_discounted"] = inventory["price"] < inventory["compare_at_price"]

discount_counts = inventory["is_discounted"].value_counts()

# Graph
discount_counts.plot(kind="bar", color=["salmon", "lightgray"], edgecolor="black")
plt.title("Products on discount")
plt.xticks([0, 1], ["Discounted", "Not discounted"], rotation=0)
plt.ylabel("Number of products")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
```


    
![png](Module_2_assesment1_files/Module_2_assesment1_17_0.png)
    


Here we can see clearly, there are a highly percentage of products that are currently discounted. 

#### *Hypothesis 5:* Some vendors dominate the inventory of the grocery


```python
top_vendors = inventory["vendor"].value_counts().head(10)

top_vendors.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Top 10 vendors")
plt.ylabel("Number of products")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```


    
![png](Module_2_assesment1_files/Module_2_assesment1_20_0.png)
    


#### *Hypothesis 6:* The grocery offers more products of an specific category than others


```python
top_types = inventory["product_type"].value_counts().head(10)

top_types.plot(kind="bar", color="mediumseagreen", edgecolor="black")
plt.title("Top 10 product types")
plt.ylabel("Number of variants")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
```


    
![png](Module_2_assesment1_files/Module_2_assesment1_22_0.png)
    


### REGULARS dataset:
#### *Hypothesis 7:* Only a small fraction of users regularly mark products, and some rely heavily on repeated purchases.


```python
regular_counts = regulars["user_id"].value_counts()

plt.figure(figsize=(8, 4))
regular_counts[regular_counts < 50].plot(kind="hist", bins=25, edgecolor="black", alpha=0.7)
plt.title("Distribution of number of regular products per user (under 50)")
plt.xlabel("Number of regular products marked")
plt.ylabel("Number of users")
plt.tight_layout()
plt.show()

```


    
![png](Module_2_assesment1_files/Module_2_assesment1_24_0.png)
    


Most of customers marked as "regulars" only one or two products. It is curious that there is no customer without any product marked as regular.

### ABANDONED_CARTS dataset:
#### *Hypothesis 8:* Abandonment patterns over time may indicate specific pain points or seasonal effects.


```python
import pandas as pd
import matplotlib.pyplot as plt

# Convertir fechas
abandoned["created_at"] = pd.to_datetime(abandoned["created_at"])

# Agrupar por fecha
abandoned_daily = (
    abandoned.groupby(abandoned["created_at"].dt.date)
    .size()
    .reset_index(name="cart_count")
)
abandoned_daily["created_at"] = pd.to_datetime(abandoned_daily["created_at"])

# Filtrar fechas desde 2021-08-01
abandoned_daily = abandoned_daily[abandoned_daily["created_at"] >= pd.to_datetime("2021-08-01")]

# Media móvil
abandoned_daily["rolling_avg"] = abandoned_daily["cart_count"].rolling(window=7).mean()

# Gráfico
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(abandoned_daily["created_at"], abandoned_daily["cart_count"],
        label="Daily Abandoned Carts", color="orange", linewidth=1, alpha=0.6)
ax.plot(abandoned_daily["created_at"], abandoned_daily["rolling_avg"],
        label="7-day Moving Average", color="black", linestyle="--", linewidth=2)

# Estética
ax.set_title("Abandoned Carts per Day with 7-day Rolling Average", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Number of Abandoned Carts")
ax.legend()
plt.tight_layout()
plt.show()

```


    
![png](Module_2_assesment1_files/Module_2_assesment1_27_0.png)
    


Gradual Increase: From August 2021 to early 2022, we observe a consistent upward trend in cart abandonments. This might reflect growing user base or potential usability issues emerging over time.
There is also a spike around mid-January 2022 reaching 250 abandoned carts. It could be a technical issue, failed promotion... After that, abandonment rates return to moderate levels but higher than initial months.

## 5) Summary and Next Steps

This exploratory analysis has revealed several important patterns:
- Most users are one-time buyers, but there is a small loyal group.
- Some products and vendors dominate the inventory.
- Cart abandonment increased significantly in late 2021, potentially indicating UX or pricing issues.

These insights could inform future marketing strategies, product selection optimization, and targeted campaigns.



```python

```
