import { useState } from 'react'

function DataScienceComplete() {
  const [activeSection, setActiveSection] = useState('introduction')
  const [expandedCode, setExpandedCode] = useState(null)

  const sections = [
    { id: 'introduction', title: 'Data Science Introduction', icon: '📊' },
    { id: 'numpy', title: 'NumPy Mastery', icon: '🔢' },
    { id: 'pandas', title: 'Pandas Complete', icon: '🐼' },
    { id: 'matplotlib', title: 'Data Visualization', icon: '📈' },
    { id: 'data-analysis', title: 'Exploratory Data Analysis', icon: '🔍' },
    { id: 'statistics', title: 'Statistical Analysis', icon: '📊' },
    { id: 'data-cleaning', title: 'Advanced Data Cleaning', icon: '🧹' },
    { id: 'time-series', title: 'Time Series Analysis', icon: '⏰' },
    { id: 'real-projects', title: 'Real-World Projects', icon: '🌍' }
  ]

  const codeExamples = {
    numpyComplete: `# Complete NumPy Tutorial: From Basics to Advanced
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

print("=== NUMPY FUNDAMENTALS ===")
print("NumPy version:", np.__version__)

# 1. ARRAY CREATION - Multiple Ways
print("\\n1. ARRAY CREATION METHODS")
print("-" * 30)

# From lists
list_1d = [1, 2, 3, 4, 5]
arr_1d = np.array(list_1d)
print(f"From list: {arr_1d}")

# From nested lists (2D)
list_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr_2d = np.array(list_2d)
print(f"2D array:\\n{arr_2d}")

# Built-in creation functions
zeros = np.zeros((3, 4))  # 3x4 array of zeros
ones = np.ones((2, 3))    # 2x3 array of ones
empty = np.empty((2, 2))  # Uninitialized array
identity = np.eye(3)      # 3x3 identity matrix

print(f"Zeros array:\\n{zeros}")
print(f"Identity matrix:\\n{identity}")

# Range arrays
arange_arr = np.arange(0, 10, 2)  # Start, stop, step
linspace_arr = np.linspace(0, 1, 5)  # Start, stop, num_points

print(f"Arange (0 to 10, step 2): {arange_arr}")
print(f"Linspace (0 to 1, 5 points): {linspace_arr}")

# Random arrays
np.random.seed(42)  # For reproducibility
random_arr = np.random.rand(3, 3)  # Uniform [0, 1)
normal_arr = np.random.randn(3, 3)  # Standard normal
randint_arr = np.random.randint(1, 10, (3, 3))  # Random integers

print(f"Random uniform:\\n{random_arr}")
print(f"Random normal:\\n{normal_arr}")

# 2. ARRAY ATTRIBUTES AND PROPERTIES
print("\\n2. ARRAY ATTRIBUTES")
print("-" * 30)

sample_arr = np.random.randint(1, 100, (3, 4, 2))
print(f"Array shape: {sample_arr.shape}")      # Dimensions
print(f"Array size: {sample_arr.size}")        # Total elements
print(f"Array ndim: {sample_arr.ndim}")        # Number of dimensions
print(f"Array dtype: {sample_arr.dtype}")      # Data type
print(f"Array itemsize: {sample_arr.itemsize}") # Bytes per element
print(f"Array nbytes: {sample_arr.nbytes}")    # Total bytes

# 3. ARRAY INDEXING AND SLICING
print("\\n3. INDEXING AND SLICING")
print("-" * 30)

# 1D indexing
arr = np.array([10, 20, 30, 40, 50])
print(f"Original array: {arr}")
print(f"First element: {arr[0]}")
print(f"Last element: {arr[-1]}")
print(f"Slice [1:4]: {arr[1:4]}")
print(f"Every other element: {arr[::2]}")

# 2D indexing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\\n2D Matrix:\\n{matrix}")
print(f"Element at (1,2): {matrix[1, 2]}")
print(f"First row: {matrix[0, :]}")
print(f"Second column: {matrix[:, 1]}")
print(f"Submatrix:\\n{matrix[0:2, 1:3]}")

# Boolean indexing
arr = np.array([1, 5, 3, 8, 2, 9])
mask = arr > 5
print(f"\\nOriginal: {arr}")
print(f"Mask (>5): {mask}")
print(f"Elements > 5: {arr[mask]}")

# Fancy indexing
indices = [0, 2, 4]
print(f"Fancy indexing {indices}: {arr[indices]}")

# 4. ARRAY OPERATIONS
print("\\n4. ARRAY OPERATIONS")
print("-" * 30)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Element-wise operations
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")  # Element-wise multiplication
print(f"a ** 2 = {a ** 2}")
print(f"np.sqrt(a) = {np.sqrt(a)}")

# Universal functions (ufuncs)
angles = np.array([0, np.pi/4, np.pi/2, np.pi])
print(f"\\nAngles: {angles}")
print(f"sin(angles): {np.sin(angles)}")
print(f"cos(angles): {np.cos(angles)}")
print(f"exp(a): {np.exp([1, 2, 3])}")
print(f"log(a): {np.log([1, np.e, np.e**2])}")

# 5. BROADCASTING
print("\\n5. BROADCASTING")
print("-" * 30)

# Scalar with array
arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
result = arr + scalar
print(f"Array:\\n{arr}")
print(f"Array + {scalar}:\\n{result}")

# Arrays with different shapes
a = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
b = np.array([10, 20, 30])             # 1x3
result = a + b  # b is broadcast to each row of a
print(f"\\nBroadcasting example:")
print(f"a (2x3):\\n{a}")
print(f"b (1x3): {b}")
print(f"a + b:\\n{result}")

# 6. ARRAY MANIPULATION
print("\\n6. ARRAY MANIPULATION")
print("-" * 30)

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original:\\n{arr}")

# Reshape
reshaped = arr.reshape(3, 2)
print(f"Reshaped (3x2):\\n{reshaped}")

# Flatten
flattened = arr.flatten()
print(f"Flattened: {flattened}")

# Transpose
transposed = arr.T
print(f"Transposed:\\n{transposed}")

# Stack arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
vstack_result = np.vstack([arr1, arr2])  # Vertical stack
hstack_result = np.hstack([arr1, arr2])  # Horizontal stack

print(f"\\nVertical stack:\\n{vstack_result}")
print(f"Horizontal stack: {hstack_result}")

# Split arrays
split_result = np.split(hstack_result, 2)
print(f"Split result: {split_result}")

# 7. STATISTICAL FUNCTIONS
print("\\n7. STATISTICAL FUNCTIONS")
print("-" * 30)

data = np.random.randn(1000)  # 1000 random numbers
print(f"Data sample (first 10): {data[:10]}")
print(f"Mean: {np.mean(data):.3f}")
print(f"Median: {np.median(data):.3f}")
print(f"Std deviation: {np.std(data):.3f}")
print(f"Variance: {np.var(data):.3f}")
print(f"Min: {np.min(data):.3f}")
print(f"Max: {np.max(data):.3f}")
print(f"25th percentile: {np.percentile(data, 25):.3f}")
print(f"75th percentile: {np.percentile(data, 75):.3f}")

# 2D statistical operations
matrix = np.random.randint(1, 10, (3, 4))
print(f"\\nMatrix:\\n{matrix}")
print(f"Sum of all elements: {np.sum(matrix)}")
print(f"Sum along axis 0 (columns): {np.sum(matrix, axis=0)}")
print(f"Sum along axis 1 (rows): {np.sum(matrix, axis=1)}")
print(f"Mean along axis 0: {np.mean(matrix, axis=0)}")

# 8. LINEAR ALGEBRA
print("\\n8. LINEAR ALGEBRA")
print("-" * 30)

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\\n{A}")
print(f"Matrix B:\\n{B}")

# Matrix multiplication
dot_product = np.dot(A, B)  # or A @ B
print(f"A · B (dot product):\\n{dot_product}")

# Matrix properties
det_A = np.linalg.det(A)
inv_A = np.linalg.inv(A)
eigenvals, eigenvecs = np.linalg.eig(A)

print(f"Determinant of A: {det_A:.3f}")
print(f"Inverse of A:\\n{inv_A}")
print(f"Eigenvalues of A: {eigenvals}")
print(f"Eigenvectors of A:\\n{eigenvecs}")

# Solving linear systems: Ax = b
b = np.array([5, 6])
x = np.linalg.solve(A, b)
print(f"\\nSolving Ax = b where b = {b}")
print(f"Solution x = {x}")
print(f"Verification Ax = {A @ x}")

# 9. ADVANCED OPERATIONS
print("\\n9. ADVANCED OPERATIONS")
print("-" * 30)

# Conditional operations
arr = np.array([1, 5, 3, 8, 2, 9])
result = np.where(arr > 5, arr, -1)  # Replace values <=5 with -1
print(f"Original: {arr}")
print(f"Where >5, keep value, else -1: {result}")

# Unique values
arr = np.array([1, 2, 2, 3, 3, 3, 4])
unique_vals = np.unique(arr)
unique_vals, counts = np.unique(arr, return_counts=True)
print(f"\\nArray with duplicates: {arr}")
print(f"Unique values: {unique_vals}")
print(f"Value counts: {counts}")

# Set operations
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([3, 4, 5, 6, 7])
intersection = np.intersect1d(arr1, arr2)
union = np.union1d(arr1, arr2)
difference = np.setdiff1d(arr1, arr2)

print(f"\\nArray 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Intersection: {intersection}")
print(f"Union: {union}")
print(f"Difference (arr1 - arr2): {difference}")

# 10. PERFORMANCE COMPARISON
print("\\n10. PERFORMANCE COMPARISON")
print("-" * 30)

# NumPy vs Python lists performance test
size = 100000

# Python lists
python_list1 = list(range(size))
python_list2 = list(range(size, 2*size))

# NumPy arrays
numpy_arr1 = np.arange(size)
numpy_arr2 = np.arange(size, 2*size)

# Time Python list addition
start_time = time.time()
python_result = [a + b for a, b in zip(python_list1, python_list2)]
python_time = time.time() - start_time

# Time NumPy array addition
start_time = time.time()
numpy_result = numpy_arr1 + numpy_arr2
numpy_time = time.time() - start_time

print(f"Array size: {size:,}")
print(f"Python lists time: {python_time:.4f} seconds")
print(f"NumPy arrays time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster!")

# 11. PRACTICAL EXAMPLES
print("\\n11. PRACTICAL EXAMPLES")
print("-" * 30)

# Example 1: Calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

stock_prices = np.array([100, 101, 99, 102, 105, 103, 107, 109, 108, 106])
ma_3 = moving_average(stock_prices, 3)
print(f"Stock prices: {stock_prices}")
print(f"3-day moving average: {ma_3}")

# Example 2: Distance calculation
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

p1 = np.array([1, 2, 3])
p2 = np.array([4, 5, 6])
distance = euclidean_distance(p1, p2)
print(f"\\nDistance between {p1} and {p2}: {distance:.3f}")

# Example 3: Normalize data (z-score normalization)
def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

raw_data = np.random.randint(1, 100, 10)
normalized = normalize_data(raw_data)
print(f"\\nRaw data: {raw_data}")
print(f"Normalized: {normalized}")
print(f"Mean after normalization: {np.mean(normalized):.6f}")
print(f"Std after normalization: {np.std(normalized):.6f}")

print("\\n" + "="*50)
print("🎉 NUMPY MASTERY COMPLETE!")
print("You now know NumPy from basics to advanced operations.")
print("Next: Apply these skills in data science projects!")
print("="*50)`,

    pandasComplete: `# Complete Pandas Tutorial: Data Manipulation and Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

print("=== PANDAS COMPLETE GUIDE ===")
print("Pandas version:", pd.__version__)

# 1. DATA STRUCTURES - Series and DataFrame
print("\\n1. PANDAS DATA STRUCTURES")
print("-" * 40)

# Series - 1D labeled array
print("SERIES (1D labeled array):")
series_from_list = pd.Series([1, 2, 3, 4, 5])
print(f"From list:\\n{series_from_list}")

series_with_index = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(f"\\nWith custom index:\\n{series_with_index}")

series_from_dict = pd.Series({'Apple': 1.5, 'Banana': 0.8, 'Orange': 1.2})
print(f"\\nFrom dictionary:\\n{series_from_dict}")

# DataFrame - 2D labeled data structure
print("\\nDATAFRAME (2D labeled data structure):")

# From dictionary
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle'],
    'Salary': [70000, 85000, 90000, 75000, 88000],
    'Experience': [2, 5, 8, 3, 6]
}

df = pd.DataFrame(data_dict)
print("DataFrame from dictionary:")
print(df)
print(f"\\nDataFrame info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Index: {list(df.index)}")

# 2. READING AND WRITING DATA
print("\\n2. DATA I/O OPERATIONS")
print("-" * 40)

# Create sample data for demonstration
np.random.seed(42)
sample_data = pd.DataFrame({
    'product_id': range(1000, 1100),
    'product_name': [f'Product_{i}' for i in range(100)],
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 100),
    'price': np.random.uniform(10, 500, 100).round(2),
    'quantity': np.random.randint(1, 100, 100),
    'rating': np.random.uniform(1, 5, 100).round(1),
    'date_added': pd.date_range('2023-01-01', periods=100, freq='D')
})

print("Sample dataset created:")
print(sample_data.head())

# Save data (examples - would actually create files)
# sample_data.to_csv('products.csv', index=False)
# sample_data.to_excel('products.xlsx', index=False)
# sample_data.to_json('products.json')

print("\\nData can be saved as:")
print("• CSV: df.to_csv('file.csv')")
print("• Excel: df.to_excel('file.xlsx')")
print("• JSON: df.to_json('file.json')")
print("• SQL: df.to_sql('table_name', connection)")

# 3. DATA EXPLORATION
print("\\n3. DATA EXPLORATION")
print("-" * 40)

df_explore = sample_data.copy()

print("BASIC INFO:")
print(f"Shape: {df_explore.shape}")
print(f"\\nColumn data types:")
print(df_explore.dtypes)

print(f"\\nBasic statistics:")
print(df_explore.describe())

print(f"\\nInfo about the dataset:")
print(df_explore.info())

print(f"\\nFirst 3 rows:")
print(df_explore.head(3))

print(f"\\nLast 3 rows:")
print(df_explore.tail(3))

print(f"\\nRandom sample of 3 rows:")
print(df_explore.sample(3))

# 4. DATA SELECTION AND INDEXING
print("\\n4. DATA SELECTION AND INDEXING")
print("-" * 40)

# Column selection
print("COLUMN SELECTION:")
print("Single column (Series):")
print(df_explore['product_name'].head(3))

print("\\nMultiple columns (DataFrame):")
print(df_explore[['product_name', 'price', 'rating']].head(3))

# Row selection
print("\\nROW SELECTION:")
print("By index position (.iloc):")
print(df_explore.iloc[0])  # First row

print("\\nBy index label (.loc):")
print(df_explore.loc[0:2, 'product_name':'price'])  # Rows 0-2, specific columns

print("\\nBoolean indexing:")
expensive_products = df_explore[df_explore['price'] > 300]
print(f"Products over \\$300: {len(expensive_products)}")
print(expensive_products[['product_name', 'price']].head(3))

# 5. DATA FILTERING
print("\\n5. DATA FILTERING")
print("-" * 40)

# Single condition
electronics = df_explore[df_explore['category'] == 'Electronics']
print(f"Electronics products: {len(electronics)}")

# Multiple conditions
high_rated_expensive = df_explore[
    (df_explore['rating'] >= 4.0) & (df_explore['price'] >= 200)
]
print(f"High-rated expensive products: {len(high_rated_expensive)}")

# Using isin() for multiple values
categories_of_interest = df_explore[
    df_explore['category'].isin(['Electronics', 'Books'])
]
print(f"Electronics and Books: {len(categories_of_interest)}")

# String operations
products_with_1 = df_explore[df_explore['product_name'].str.contains('1')]
print(f"Products with '1' in name: {len(products_with_1)}")

# 6. DATA SORTING
print("\\n6. DATA SORTING")
print("-" * 40)

# Sort by single column
sorted_by_price = df_explore.sort_values('price', ascending=False)
print("Top 3 most expensive products:")
print(sorted_by_price[['product_name', 'price']].head(3))

# Sort by multiple columns
sorted_multi = df_explore.sort_values(['category', 'price'], ascending=[True, False])
print("\\nSorted by category (asc) then price (desc):")
print(sorted_multi[['product_name', 'category', 'price']].head(5))

# 7. GROUPING AND AGGREGATION
print("\\n7. GROUPING AND AGGREGATION")
print("-" * 40)

# Group by single column
category_stats = df_explore.groupby('category').agg({
    'price': ['mean', 'min', 'max', 'std'],
    'rating': 'mean',
    'quantity': 'sum'
}).round(2)

print("Statistics by category:")
print(category_stats)

# Multiple grouping
# Add a price range category for demonstration
df_explore['price_range'] = pd.cut(df_explore['price'], 
                                 bins=[0, 50, 150, 300, 500], 
                                 labels=['Low', 'Medium', 'High', 'Premium'])

price_category_stats = df_explore.groupby(['category', 'price_range']).size()
print("\\nProduct count by category and price range:")
print(price_category_stats)

# 8. DATA TRANSFORMATION
print("\\n8. DATA TRANSFORMATION")
print("-" * 40)

df_transform = df_explore.copy()

# Add new columns
df_transform['total_value'] = df_transform['price'] * df_transform['quantity']
df_transform['price_per_rating'] = df_transform['price'] / df_transform['rating']
df_transform['is_expensive'] = df_transform['price'] > df_transform['price'].median()

print("New columns added:")
print(df_transform[['product_name', 'price', 'rating', 'total_value', 'is_expensive']].head(3))

# Apply functions
def categorize_rating(rating):
    if rating >= 4.5:
        return 'Excellent'
    elif rating >= 4.0:
        return 'Good'
    elif rating >= 3.0:
        return 'Average'
    else:
        return 'Poor'

df_transform['rating_category'] = df_transform['rating'].apply(categorize_rating)
print("\\nRating categories:")
print(df_transform['rating_category'].value_counts())

# Lambda functions
df_transform['name_length'] = df_transform['product_name'].apply(lambda x: len(x))
print(f"\\nAverage product name length: {df_transform['name_length'].mean():.1f}")

# 9. HANDLING MISSING DATA
print("\\n9. HANDLING MISSING DATA")
print("-" * 40)

# Create data with missing values for demonstration
df_missing = df_transform.copy()
np.random.seed(42)
missing_indices = np.random.choice(df_missing.index, size=10, replace=False)
df_missing.loc[missing_indices, 'rating'] = np.nan
df_missing.loc[missing_indices[:5], 'price'] = np.nan

print(f"Missing values per column:")
print(df_missing.isnull().sum())

# Different strategies for handling missing data
print("\\nStrategies for handling missing data:")

# 1. Drop rows with any missing values
df_dropped = df_missing.dropna()
print(f"After dropping rows with any NaN: {df_dropped.shape[0]} rows")

# 2. Drop rows with missing values in specific columns
df_dropped_specific = df_missing.dropna(subset=['price', 'rating'])
print(f"After dropping rows with NaN in price/rating: {df_dropped_specific.shape[0]} rows")

# 3. Fill missing values
df_filled = df_missing.copy()
df_filled['price'].fillna(df_filled['price'].median(), inplace=True)
df_filled['rating'].fillna(df_filled['rating'].mean(), inplace=True)
print(f"After filling missing values: {df_filled.isnull().sum()['price']} price NaN, {df_filled.isnull().sum()['rating']} rating NaN")

# 10. DATA MERGING AND JOINING
print("\\n10. DATA MERGING AND JOINING")
print("-" * 40)

# Create additional datasets for merging examples
suppliers = pd.DataFrame({
    'product_id': [1001, 1003, 1005, 1007, 1009],
    'supplier_name': ['Supplier_A', 'Supplier_B', 'Supplier_C', 'Supplier_D', 'Supplier_E'],
    'supplier_location': ['USA', 'China', 'Germany', 'Japan', 'India']
})

reviews = pd.DataFrame({
    'product_id': [1001, 1001, 1003, 1003, 1003, 1005],
    'review_score': [5, 4, 5, 3, 4, 5],
    'review_text': ['Great!', 'Good', 'Excellent', 'OK', 'Nice', 'Perfect']
})

print("Suppliers data:")
print(suppliers)

print("\\nReviews data:")
print(reviews)

# Inner join
inner_merged = pd.merge(df_explore[['product_id', 'product_name', 'price']], 
                       suppliers, 
                       on='product_id', 
                       how='inner')
print(f"\\nInner join result: {len(inner_merged)} rows")
print(inner_merged)

# Left join
left_merged = pd.merge(df_explore[['product_id', 'product_name', 'price']].head(10), 
                      suppliers, 
                      on='product_id', 
                      how='left')
print(f"\\nLeft join result (first 10 products): {len(left_merged)} rows")
print(left_merged[['product_name', 'supplier_name']])

# 11. TIME SERIES OPERATIONS
print("\\n11. TIME SERIES OPERATIONS")
print("-" * 40)

# Create time series data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'sales': np.random.randint(100, 1000, 365) + np.sin(np.arange(365) * 2 * np.pi / 30) * 50,
    'temperature': 20 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365) * 3
}).round(2)

ts_data.set_index('date', inplace=True)
print("Time series data (first 5 days):")
print(ts_data.head())

# Time-based selection
january_data = ts_data['2023-01']
print(f"\\nJanuary data: {len(january_data)} days")
print(january_data.head(3))

# Resampling
monthly_avg = ts_data.resample('M').mean()
print("\\nMonthly averages:")
print(monthly_avg.head(3))

# Rolling operations
ts_data['sales_7day_avg'] = ts_data['sales'].rolling(window=7).mean()
ts_data['sales_30day_std'] = ts_data['sales'].rolling(window=30).std()

print("\\nWith rolling statistics:")
print(ts_data[['sales', 'sales_7day_avg', 'sales_30day_std']].head(35).tail(5))

# 12. ADVANCED OPERATIONS
print("\\n12. ADVANCED OPERATIONS")
print("-" * 40)

# Pivot tables
pivot_data = df_explore.copy()
pivot_table = pivot_data.pivot_table(
    values='price',
    index='category',
    columns='price_range',
    aggfunc=['mean', 'count'],
    fill_value=0
).round(2)

print("Pivot table (mean price by category and price range):")
print(pivot_table)

# Cross-tabulation
crosstab = pd.crosstab(df_explore['category'], df_explore['price_range'], margins=True)
print("\\nCross-tabulation:")
print(crosstab)

# String operations
df_explore['category_upper'] = df_explore['category'].str.upper()
df_explore['category_length'] = df_explore['category'].str.len()
df_explore['first_letter'] = df_explore['product_name'].str[0]

print("\\nString operations:")
print(df_explore[['category', 'category_upper', 'category_length', 'first_letter']].head(3))

# 13. DATA VALIDATION AND CLEANING
print("\\n13. DATA VALIDATION AND CLEANING")
print("-" * 40)

# Detect duplicates
print(f"Duplicate rows: {df_explore.duplicated().sum()}")

# Detect outliers using IQR
Q1 = df_explore['price'].quantile(0.25)
Q3 = df_explore['price'].quantile(0.75)
IQR = Q3 - Q1
outliers = df_explore[(df_explore['price'] < Q1 - 1.5*IQR) | 
                     (df_explore['price'] > Q3 + 1.5*IQR)]
print(f"Price outliers (IQR method): {len(outliers)}")

# Data type conversions
df_types = df_explore.copy()
print(f"\\nOriginal data types:")
print(df_types.dtypes)

# Convert to appropriate types
df_types['product_id'] = df_types['product_id'].astype('str')
df_types['category'] = df_types['category'].astype('category')
print(f"\\nAfter conversion:")
print(df_types.dtypes)

# 14. PERFORMANCE TIPS
print("\\n14. PERFORMANCE TIPS")
print("-" * 40)

# Use vectorized operations instead of loops
print("Performance comparison: Loop vs Vectorized")

# Slow way (don't do this)
import time
start_time = time.time()
slow_result = []
for _, row in df_explore.head(1000).iterrows():
    slow_result.append(row['price'] * row['quantity'])
slow_time = time.time() - start_time

# Fast way (vectorized)
start_time = time.time()
fast_result = df_explore.head(1000)['price'] * df_explore.head(1000)['quantity']
fast_time = time.time() - start_time

print(f"Loop time: {slow_time:.4f} seconds")
print(f"Vectorized time: {fast_time:.4f} seconds")
print(f"Vectorized is {slow_time/fast_time:.1f}x faster!")

# Use categorical data for memory efficiency
memory_before = df_explore['category'].memory_usage(deep=True)
df_explore['category'] = df_explore['category'].astype('category')
memory_after = df_explore['category'].memory_usage(deep=True)
print(f"\\nMemory usage - Before: {memory_before} bytes, After: {memory_after} bytes")
print(f"Memory saved: {((memory_before - memory_after) / memory_before * 100):.1f}%")

print("\\n" + "="*60)
print("🐼 PANDAS MASTERY COMPLETE!")
print("You now have comprehensive knowledge of pandas:")
print("• Data structures (Series, DataFrame)")
print("• Data I/O operations") 
print("• Data exploration and selection")
print("• Filtering, sorting, and grouping")
print("• Data transformation and cleaning")
print("• Merging and joining datasets")
print("• Time series analysis")
print("• Advanced operations and performance optimization")
print("\\nNext: Apply these skills to real-world data analysis projects!")
print("="*60)`,

    dataVisualization: `# Complete Data Visualization Guide with Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== DATA VISUALIZATION MASTERY ===")
print("Matplotlib version:", plt.matplotlib.__version__)
print("Seaborn version:", sns.__version__)

# Create comprehensive sample dataset
np.random.seed(42)
n_samples = 1000

# Generate realistic dataset
data = pd.DataFrame({
    'age': np.random.normal(35, 12, n_samples).astype(int),
    'income': np.random.lognormal(10, 0.8, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'experience': np.random.exponential(5, n_samples),
    'performance_score': np.random.beta(2, 5, n_samples) * 100,
    'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
    'satisfaction': np.random.uniform(1, 10, n_samples),
    'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle'], n_samples)
})

# Add some correlations to make data more realistic
data['income'] = data['income'] * (1 + data['experience'] * 0.1) * (1 + (data['education'].map({'High School': 0, 'Bachelor': 0.2, 'Master': 0.4, 'PhD': 0.6})))
data['performance_score'] = data['performance_score'] + data['experience'] * 2 + np.random.normal(0, 5, n_samples)
data['satisfaction'] = data['satisfaction'] + data['performance_score'] * 0.02 + np.random.normal(0, 1, n_samples)

# Clip values to reasonable ranges
data['age'] = np.clip(data['age'], 18, 65)
data['income'] = np.clip(data['income'], 20000, 500000)
data['performance_score'] = np.clip(data['performance_score'], 0, 100)
data['satisfaction'] = np.clip(data['satisfaction'], 1, 10)
data['experience'] = np.clip(data['experience'], 0, 40)

print("Dataset created with shape:", data.shape)
print("\\nFirst few rows:")
print(data.head())

# 1. BASIC PLOTS WITH MATPLOTLIB
print("\\n1. BASIC MATPLOTLIB PLOTS")
print("-" * 40)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Basic Matplotlib Plots', fontsize=16, fontweight='bold')

# 1.1 Line Plot
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

axes[0, 0].plot(x, y1, label='sin(x)', linewidth=2, color='blue')
axes[0, 0].plot(x, y2, label='cos(x)', linewidth=2, color='red', linestyle='--')
axes[0, 0].set_title('Line Plot')
axes[0, 0].set_xlabel('X values')
axes[0, 0].set_ylabel('Y values')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 1.2 Scatter Plot
axes[0, 1].scatter(data['age'], data['income'], alpha=0.6, c=data['performance_score'], cmap='viridis')
axes[0, 1].set_title('Age vs Income (colored by Performance)')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Income ($)')

# 1.3 Histogram
axes[0, 2].hist(data['income'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 2].set_title('Income Distribution')
axes[0, 2].set_xlabel('Income ($)')
axes[0, 2].set_ylabel('Frequency')

# 1.4 Bar Plot
education_counts = data['education'].value_counts()
axes[1, 0].bar(education_counts.index, education_counts.values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
axes[1, 0].set_title('Education Level Distribution')
axes[1, 0].set_xlabel('Education Level')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# 1.5 Box Plot
dept_performance = [data[data['department'] == dept]['performance_score'] for dept in data['department'].unique()]
axes[1, 1].boxplot(dept_performance, labels=data['department'].unique())
axes[1, 1].set_title('Performance Score by Department')
axes[1, 1].set_xlabel('Department')
axes[1, 1].set_ylabel('Performance Score')
axes[1, 1].tick_params(axis='x', rotation=45)

# 1.6 Pie Chart
dept_counts = data['department'].value_counts()
axes[1, 2].pie(dept_counts.values, labels=dept_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 2].set_title('Department Distribution')

plt.tight_layout()
plt.show()

# 2. ADVANCED MATPLOTLIB CUSTOMIZATION
print("\\n2. ADVANCED MATPLOTLIB CUSTOMIZATION")
print("-" * 40)

# Create a sophisticated multi-plot figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 2], width_ratios=[2, 1, 2])

# Main scatter plot with regression line
ax_main = fig.add_subplot(gs[0, :2])
scatter = ax_main.scatter(data['experience'], data['income'], 
                         c=data['satisfaction'], s=data['performance_score']*2,
                         alpha=0.6, cmap='plasma', edgecolors='black', linewidth=0.5)

# Add regression line
z = np.polyfit(data['experience'], data['income'], 1)
p = np.poly1d(z)
ax_main.plot(data['experience'], p(data['experience']), "r--", alpha=0.8, linewidth=2)

ax_main.set_xlabel('Years of Experience', fontsize=12, fontweight='bold')
ax_main.set_ylabel('Income ($)', fontsize=12, fontweight='bold')
ax_main.set_title('Income vs Experience (Size=Performance, Color=Satisfaction)', 
                 fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax_main)
cbar.set_label('Satisfaction Score', fontsize=10)

# Side histogram for experience
ax_hist_y = fig.add_subplot(gs[0, 2])
ax_hist_y.hist(data['experience'], bins=20, orientation='horizontal', alpha=0.7, color='lightcoral')
ax_hist_y.set_xlabel('Count')
ax_hist_y.set_title('Experience\\nDistribution', fontsize=10)

# Bottom histogram for income
ax_hist_x = fig.add_subplot(gs[1, :2])
ax_hist_x.hist(data['income'], bins=30, alpha=0.7, color='lightblue')
ax_hist_x.set_xlabel('Income ($)')
ax_hist_x.set_ylabel('Count')
ax_hist_x.set_title('Income Distribution', fontsize=10)

# Correlation heatmap
ax_corr = fig.add_subplot(gs[2, :])
numeric_cols = ['age', 'income', 'experience', 'performance_score', 'satisfaction']
corr_matrix = data[numeric_cols].corr()

im = ax_corr.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax_corr.set_xticks(range(len(numeric_cols)))
ax_corr.set_yticks(range(len(numeric_cols)))
ax_corr.set_xticklabels(numeric_cols, rotation=45)
ax_corr.set_yticklabels(numeric_cols)
ax_corr.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# Add correlation values
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax_corr.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax_corr, shrink=0.8)
plt.tight_layout()
plt.show()

# 3. SEABORN STATISTICAL PLOTS
print("\\n3. SEABORN STATISTICAL PLOTS")
print("-" * 40)

# Create comprehensive seaborn visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Seaborn Statistical Visualizations', fontsize=16, fontweight='bold')

# 3.1 Distribution plots
sns.histplot(data=data, x='income', hue='education', multiple='stack', ax=axes[0, 0])
axes[0, 0].set_title('Income Distribution by Education')

# 3.2 Box plot with swarm
sns.boxplot(data=data, x='department', y='performance_score', ax=axes[0, 1])
sns.swarmplot(data=data, x='department', y='performance_score', 
              color='red', alpha=0.6, size=3, ax=axes[0, 1])
axes[0, 1].set_title('Performance Score by Department')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3.3 Violin plot
sns.violinplot(data=data, x='education', y='satisfaction', ax=axes[0, 2])
axes[0, 2].set_title('Satisfaction by Education Level')
axes[0, 2].tick_params(axis='x', rotation=45)

# 3.4 Regression plot
sns.regplot(data=data, x='experience', y='income', ax=axes[1, 0], scatter_kws={'alpha':0.6})
axes[1, 0].set_title('Experience vs Income (with regression)')

# 3.5 Heatmap
pivot_data = data.pivot_table(values='performance_score', 
                             index='education', 
                             columns='department', 
                             aggfunc='mean')
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 1])
axes[1, 1].set_title('Avg Performance by Education & Department')

# 3.6 Pair plot (subset for performance)
subset_data = data[['age', 'income', 'experience', 'satisfaction']].sample(200)
# Create pair plot separately due to size
g = sns.PairGrid(subset_data, height=2)
g.map_upper(sns.scatterplot, alpha=0.6)
g.map_lower(sns.regplot, scatter_kws={'alpha':0.6})
g.map_diag(sns.histplot)
plt.suptitle('Pair Plot of Key Variables', y=1.02)
plt.show()

# Continue with seaborn plots
# 3.7 Count plot
sns.countplot(data=data, x='city', hue='education', ax=axes[1, 2])
axes[1, 2].set_title('Education Distribution by City')
axes[1, 2].tick_params(axis='x', rotation=45)

# 3.8 Strip plot
sns.stripplot(data=data, x='department', y='age', 
              hue='education', dodge=True, alpha=0.7, ax=axes[2, 0])
axes[2, 0].set_title('Age Distribution by Department & Education')
axes[2, 0].tick_params(axis='x', rotation=45)

# 3.9 Joint plot (create separately)
# We'll create a simple scatter instead
axes[2, 1].scatter(data['performance_score'], data['satisfaction'], alpha=0.6)
axes[2, 1].set_xlabel('Performance Score')
axes[2, 1].set_ylabel('Satisfaction')
axes[2, 1].set_title('Performance vs Satisfaction')

# 3.10 Ridge plot simulation with multiple histograms
departments = data['department'].unique()
for i, dept in enumerate(departments):
    dept_data = data[data['department'] == dept]['income']
    axes[2, 2].hist(dept_data, alpha=0.6, label=dept, bins=20)
axes[2, 2].set_xlabel('Income')
axes[2, 2].set_ylabel('Frequency')
axes[2, 2].set_title('Income Distribution by Department')
axes[2, 2].legend()

plt.tight_layout()
plt.show()

# 4. ADVANCED VISUALIZATIONS
print("\\n4. ADVANCED VISUALIZATION TECHNIQUES")
print("-" * 40)

# 4.1 3D Plot
fig = plt.figure(figsize=(15, 5))

# 3D Scatter
ax1 = fig.add_subplot(131, projection='3d')
scatter = ax1.scatter(data['age'], data['experience'], data['income'], 
                     c=data['satisfaction'], cmap='viridis', alpha=0.6)
ax1.set_xlabel('Age')
ax1.set_ylabel('Experience')
ax1.set_zlabel('Income')
ax1.set_title('3D Scatter: Age, Experience, Income')
plt.colorbar(scatter, ax=ax1, shrink=0.8)

# 4.2 Contour Plot
ax2 = fig.add_subplot(132)
# Create a grid for contour plot
age_range = np.linspace(data['age'].min(), data['age'].max(), 50)
exp_range = np.linspace(data['experience'].min(), data['experience'].max(), 50)
Age, Exp = np.meshgrid(age_range, exp_range)

# Calculate average income for each age-experience combination
Income = np.zeros_like(Age)
for i in range(Age.shape[0]):
    for j in range(Age.shape[1]):
        # Find closest data points
        age_close = np.abs(data['age'] - Age[i, j]) < 2
        exp_close = np.abs(data['experience'] - Exp[i, j]) < 1
        close_points = age_close & exp_close
        if close_points.any():
            Income[i, j] = data[close_points]['income'].mean()
        else:
            Income[i, j] = data['income'].mean()

contour = ax2.contour(Age, Exp, Income, levels=10)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('Age')
ax2.set_ylabel('Experience')
ax2.set_title('Income Contour Plot')

# 4.3 Subplots with different scales
ax3 = fig.add_subplot(133)
ax3_twin = ax3.twinx()

# Group data by department for line plot
dept_avg = data.groupby('department').agg({
    'performance_score': 'mean',
    'satisfaction': 'mean'
}).reset_index()

x_pos = range(len(dept_avg))
ax3.bar(x_pos, dept_avg['performance_score'], alpha=0.7, color='skyblue', label='Performance')
ax3_twin.plot(x_pos, dept_avg['satisfaction'], 'ro-', linewidth=2, label='Satisfaction')

ax3.set_xlabel('Department')
ax3.set_ylabel('Performance Score', color='blue')
ax3_twin.set_ylabel('Satisfaction', color='red')
ax3.set_title('Performance & Satisfaction by Department')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(dept_avg['department'], rotation=45)

# Add legends
ax3.legend(loc='upper left')
ax3_twin.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 5. INTERACTIVE-STYLE PLOTS
print("\\n5. INTERACTIVE-STYLE VISUALIZATIONS")
print("-" * 40)

# Create dashboard-style visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Employee Analytics Dashboard', fontsize=16, fontweight='bold')

# 5.1 KPI Cards simulation with bar charts
kpis = {
    'Avg Performance': data['performance_score'].mean(),
    'Avg Satisfaction': data['satisfaction'].mean(),
    'Avg Income': data['income'].mean(),
    'Avg Experience': data['experience'].mean()
}

bars = axes[0, 0].bar(range(len(kpis)), list(kpis.values()), 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
axes[0, 0].set_xticks(range(len(kpis)))
axes[0, 0].set_xticklabels(list(kpis.keys()), rotation=45)
axes[0, 0].set_title('Key Performance Indicators')

# Add value labels on bars
for bar, value in zip(bars, kpis.values()):
    if 'Income' in str(value):
        label = f'\\$\\{value:,.0f\\}'
    else:
        label = f'\\{value:.1f\\}'
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(kpis.values())*0.01,
                   label, ha='center', va='bottom', fontweight='bold')

# 5.2 Department breakdown
dept_stats = data.groupby('department').agg({
    'performance_score': 'mean',
    'income': 'mean',
    'satisfaction': 'mean'
})

dept_stats.plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Department Performance Metrics')
axes[0, 1].set_xlabel('Department')
axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].tick_params(axis='x', rotation=45)

# 5.3 Education impact
edu_performance = data.groupby('education')['performance_score'].mean().sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(edu_performance)))
axes[1, 0].barh(edu_performance.index, edu_performance.values, color=colors)
axes[1, 0].set_title('Performance Score by Education Level')
axes[1, 0].set_xlabel('Average Performance Score')

# 5.4 Satisfaction vs Performance scatter with trend
high_performers = data[data['performance_score'] > data['performance_score'].quantile(0.75)]
low_performers = data[data['performance_score'] < data['performance_score'].quantile(0.25)]

axes[1, 1].scatter(low_performers['performance_score'], low_performers['satisfaction'], 
                  alpha=0.6, c='red', label='Low Performers', s=50)
axes[1, 1].scatter(high_performers['performance_score'], high_performers['satisfaction'], 
                  alpha=0.6, c='green', label='High Performers', s=50)

# Add trend line
z = np.polyfit(data['performance_score'], data['satisfaction'], 1)
p = np.poly1d(z)
x_trend = np.linspace(data['performance_score'].min(), data['performance_score'].max(), 100)
axes[1, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

axes[1, 1].set_xlabel('Performance Score')
axes[1, 1].set_ylabel('Satisfaction')
axes[1, 1].set_title('Performance vs Satisfaction Analysis')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. CUSTOMIZATION AND STYLING
print("\\n6. PLOT CUSTOMIZATION AND STYLING")
print("-" * 40)

# Create a beautifully styled plot
plt.figure(figsize=(12, 8))

# Custom color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

# Create subplot
ax = plt.subplot(111)

# Scatter plot with custom styling
for i, dept in enumerate(data['department'].unique()):
    dept_data = data[data['department'] == dept]
    plt.scatter(dept_data['experience'], dept_data['income'], 
               c=colors[i], label=dept, alpha=0.7, s=80, edgecolors='white', linewidth=1)

# Customize the plot
plt.xlabel('Years of Experience', fontsize=14, fontweight='bold')
plt.ylabel('Annual Income ($)', fontsize=14, fontweight='bold')
plt.title('Employee Income Analysis by Department', fontsize=16, fontweight='bold', pad=20)

# Custom legend
legend = plt.legend(title='Department', title_fontsize=12, fontsize=11, 
                   frameon=True, fancybox=True, shadow=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Grid and styling
plt.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

# Format y-axis to show currency
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'\\$\\{x:,.0f\\}'))

# Add subtle background color
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
plt.show()

print("\\n" + "="*60)
print("📊 DATA VISUALIZATION MASTERY COMPLETE!")
print("You now have comprehensive skills in:")
print("• Basic and advanced matplotlib plotting")
print("• Statistical visualizations with seaborn")
print("• 3D plots and contour plots")
print("• Dashboard-style analytics visualizations")
print("• Custom styling and professional presentation")
print("• Interactive-style plots for data exploration")
print("\\nNext: Apply these visualization skills to real data analysis projects!")
print("="*60)`,
  }

  const renderContent = () => {
    switch(activeSection) {
      case 'introduction':
        return (
          <div className="section-content">
            <h2>📊 Data Science Introduction</h2>
            
            <div className="intro-section">
              <h3>What is Data Science?</h3>
              <p>
                Data Science is an interdisciplinary field that combines statistics, programming, domain expertise, 
                and communication skills to extract meaningful insights from data. It's the art and science of 
                uncovering hidden patterns, trends, and knowledge from raw data to make informed decisions.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🔍 Data Science as Detective Work</h4>
              <p>
                Think of data scientists as digital detectives. Just like detectives gather evidence (data), 
                analyze clues (patterns), use tools (Python, statistics), and present findings (visualizations) 
                to solve cases (business problems), data scientists follow a similar investigative process to 
                uncover insights hidden in data.
              </p>
            </div>

            <div className="concept-deep-dive">
              <h3>🔄 The Data Science Process</h3>
              <div className="translation-process">
                <div className="step">
                  <div className="step-number">1</div>
                  <div>
                    <strong>Problem Definition</strong>
                    <p>Define clear, answerable questions that drive business value</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">2</div>
                  <div>
                    <strong>Data Collection</strong>
                    <p>Gather relevant data from various sources (databases, APIs, files)</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">3</div>
                  <div>
                    <strong>Data Exploration</strong>
                    <p>Understand data structure, quality, and initial patterns</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">4</div>
                  <div>
                    <strong>Data Cleaning</strong>
                    <p>Handle missing values, outliers, and inconsistencies</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">5</div>
                  <div>
                    <strong>Analysis & Modeling</strong>
                    <p>Apply statistical methods and machine learning algorithms</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">6</div>
                  <div>
                    <strong>Communication</strong>
                    <p>Present findings through visualizations and storytelling</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="data-types-grid">
              <div className="data-type-card">
                <h4>📊 Descriptive Analytics</h4>
                <p>What happened?</p>
                <div className="real-example">Historical sales reports, website traffic summaries, demographic breakdowns</div>
              </div>
              <div className="data-type-card">
                <h4>🔍 Diagnostic Analytics</h4>
                <p>Why did it happen?</p>
                <div className="real-example">Root cause analysis, correlation studies, A/B test results</div>
              </div>
              <div className="data-type-card">
                <h4>🔮 Predictive Analytics</h4>
                <p>What will happen?</p>
                <div className="real-example">Sales forecasting, customer churn prediction, risk assessment</div>
              </div>
              <div className="data-type-card">
                <h4>💡 Prescriptive Analytics</h4>
                <p>What should we do?</p>
                <div className="real-example">Recommendation engines, optimization strategies, decision support</div>
              </div>
            </div>
          </div>
        )

      case 'numpy':
        return (
          <div className="section-content">
            <h2>🔢 NumPy Mastery</h2>
            
            <div className="intro-section">
              <h3>Why NumPy is Essential for Data Science</h3>
              <p>
                NumPy (Numerical Python) is the foundation of the Python data science ecosystem. It provides 
                fast, memory-efficient multidimensional arrays and mathematical functions. Every major data 
                science library (Pandas, Scikit-learn, TensorFlow) is built on top of NumPy.
              </p>
            </div>

            <div className="code-example">
              <h4>Complete NumPy Tutorial: From Basics to Advanced</h4>
              <pre>{codeExamples.numpyComplete}</pre>
              <button onClick={() => setExpandedCode(expandedCode === 'numpy' ? null : 'numpy')}>
                {expandedCode === 'numpy' ? 'Hide NumPy Benefits' : 'Show Why NumPy is So Powerful'}
              </button>
              {expandedCode === 'numpy' && (
                <div className="code-explanation">
                  <h5>🚀 NumPy Advantages:</h5>
                  <ul>
                    <li><strong>Performance:</strong> 10-100x faster than pure Python for numerical operations</li>
                    <li><strong>Memory Efficiency:</strong> Stores data in contiguous memory blocks</li>
                    <li><strong>Vectorization:</strong> Apply operations to entire arrays without loops</li>
                    <li><strong>Broadcasting:</strong> Perform operations on arrays with different shapes</li>
                    <li><strong>Ecosystem:</strong> Foundation for all major data science libraries</li>
                  </ul>
                </div>
              )}
            </div>

            <div className="concept-deep-dive">
              <h3>⚡ NumPy Core Concepts</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>📦 N-dimensional Arrays</h4>
                  <p>Efficient storage and manipulation of homogeneous data</p>
                  <div className="real-example">Images (3D), time series (1D), matrices (2D)</div>
                </div>
                <div className="data-type-card">
                  <h4>🔄 Broadcasting</h4>
                  <p>Perform operations on arrays with different shapes</p>
                  <div className="real-example">Add scalar to matrix, combine different sized arrays</div>
                </div>
                <div className="data-type-card">
                  <h4>⚡ Vectorization</h4>
                  <p>Apply functions to entire arrays without explicit loops</p>
                  <div className="real-example">Mathematical operations, data transformations</div>
                </div>
                <div className="data-type-card">
                  <h4>🧮 Linear Algebra</h4>
                  <p>Matrix operations, decompositions, and solving equations</p>
                  <div className="real-example">Machine learning algorithms, data transformations</div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'pandas':
        return (
          <div className="section-content">
            <h2>🐼 Pandas Complete</h2>
            
            <div className="intro-section">
              <h3>Master Data Manipulation with Pandas</h3>
              <p>
                Pandas is the most important library for data manipulation and analysis in Python. It provides 
                powerful data structures (Series and DataFrame) and data analysis tools that make working with 
                structured data fast, easy, and expressive. Think of it as Excel for Python, but much more powerful.
              </p>
            </div>

            <div className="code-example">
              <h4>Complete Pandas Guide: From Basics to Advanced</h4>
              <pre>{codeExamples.pandasComplete}</pre>
            </div>

            <div className="concept-deep-dive">
              <h3>🔧 Pandas Core Capabilities</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>📊 Data Structures</h4>
                  <p>Series (1D) and DataFrame (2D) for labeled data</p>
                  <div className="real-example">Time series, spreadsheet data, database tables</div>
                </div>
                <div className="data-type-card">
                  <h4>🔄 Data I/O</h4>
                  <p>Read/write data from various formats</p>
                  <div className="real-example">CSV, Excel, JSON, SQL, Parquet, HTML</div>
                </div>
                <div className="data-type-card">
                  <h4>🧹 Data Cleaning</h4>
                  <p>Handle missing values, duplicates, and data types</p>
                  <div className="real-example">Fill missing values, remove duplicates, convert types</div>
                </div>
                <div className="data-type-card">
                  <h4>📈 Data Analysis</h4>
                  <p>Group by, aggregation, pivot tables, and time series</p>
                  <div className="real-example">Sales analysis, customer segmentation, trend analysis</div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'matplotlib':
        return (
          <div className="section-content">
            <h2>📈 Data Visualization</h2>
            
            <div className="intro-section">
              <h3>Master Data Visualization</h3>
              <p>
                Data visualization is crucial for understanding patterns, communicating insights, and making 
                data-driven decisions. Matplotlib provides the foundation for creating static, animated, and 
                interactive visualizations in Python, while Seaborn adds statistical plotting capabilities 
                with beautiful default styles.
              </p>
            </div>

            <div className="code-example">
              <h4>Complete Data Visualization Guide</h4>
              <pre>{codeExamples.dataVisualization}</pre>
            </div>

            <div className="concept-deep-dive">
              <h3>📊 Visualization Types</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>📊 Distribution Plots</h4>
                  <p>Understand data distributions and patterns</p>
                  <div className="real-example">Histograms, box plots, violin plots, density plots</div>
                </div>
                <div className="data-type-card">
                  <h4>🔗 Relationship Plots</h4>
                  <p>Explore correlations and relationships</p>
                  <div className="real-example">Scatter plots, regression lines, pair plots, heatmaps</div>
                </div>
                <div className="data-type-card">
                  <h4>📊 Categorical Plots</h4>
                  <p>Compare categories and groups</p>
                  <div className="real-example">Bar charts, count plots, strip plots, swarm plots</div>
                </div>
                <div className="data-type-card">
                  <h4>⏰ Time Series Plots</h4>
                  <p>Visualize trends over time</p>
                  <div className="real-example">Line charts, area plots, seasonal decomposition</div>
                </div>
              </div>
            </div>
          </div>
        )

      default:
        return <div>Select a topic from the sidebar to begin your data science journey!</div>
    }
  }

  return (
    <div className="page">
      <div className="learning-container">
        <div className="sidebar">
          <h3>📊 Data Science Mastery</h3>
          <div className="section-nav">
            {sections.map((section) => (
              <button
                key={section.id}
                className={`section-link ${activeSection === section.id ? 'active' : ''}`}
                onClick={() => setActiveSection(section.id)}
              >
                {section.icon} {section.title}
              </button>
            ))}
          </div>
        </div>

        <div className="content-area">
          <div className="section-header">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${((sections.findIndex(s => s.id === activeSection) + 1) / sections.length) * 100}%` }}
              ></div>
            </div>
            <p>Section {sections.findIndex(s => s.id === activeSection) + 1} of {sections.length}</p>
          </div>

          {renderContent()}

          <div className="section-navigation">
            <button 
              className="nav-btn"
              onClick={() => {
                const currentIndex = sections.findIndex(s => s.id === activeSection)
                if (currentIndex > 0) {
                  setActiveSection(sections[currentIndex - 1].id)
                }
              }}
              disabled={sections.findIndex(s => s.id === activeSection) === 0}
            >
              ← Previous
            </button>
            <button 
              className="nav-btn"
              onClick={() => {
                const currentIndex = sections.findIndex(s => s.id === activeSection)
                if (currentIndex < sections.length - 1) {
                  setActiveSection(sections[currentIndex + 1].id)
                }
              }}
              disabled={sections.findIndex(s => s.id === activeSection) === sections.length - 1}
            >
              Next →
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DataScienceComplete