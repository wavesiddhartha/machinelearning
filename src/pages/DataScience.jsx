function DataScience() {
  return (
    <div className="page">
      <div className="content">
        <h1>🔬 Data Science with Python</h1>
        
        <section className="ds-intro">
          <h2>The Data Science Stack</h2>
          <p>Data science combines domain expertise, programming skills, and mathematical knowledge to extract insights from data. Python's ecosystem provides powerful tools for every step of the data science pipeline.</p>
          
          <div className="stack-overview">
            <div className="stack-item">
              <h3>📊 NumPy</h3>
              <p>Numerical computing foundation</p>
            </div>
            <div className="stack-item">
              <h3>🐼 Pandas</h3>
              <p>Data manipulation and analysis</p>
            </div>
            <div className="stack-item">
              <h3>📈 Matplotlib</h3>
              <p>Data visualization</p>
            </div>
            <div className="stack-item">
              <h3>📊 Seaborn</h3>
              <p>Statistical data visualization</p>
            </div>
            <div className="stack-item">
              <h3>🧮 SciPy</h3>
              <p>Scientific computing</p>
            </div>
            <div className="stack-item">
              <h3>🤖 Scikit-learn</h3>
              <p>Machine learning</p>
            </div>
          </div>
        </section>

        <section className="numpy-section">
          <h2>🔢 NumPy - Numerical Python</h2>
          <p>NumPy is the foundation of the Python data science ecosystem. It provides efficient arrays and mathematical functions that are essential for machine learning.</p>
          
          <h3>Why NumPy?</h3>
          <ul>
            <li><strong>Performance:</strong> 10-100x faster than pure Python</li>
            <li><strong>Memory Efficient:</strong> Stores data in contiguous memory</li>
            <li><strong>Vectorization:</strong> Apply operations to entire arrays</li>
            <li><strong>Broadcasting:</strong> Operate on arrays of different shapes</li>
          </ul>

          <div className="code-example">
            <h4>NumPy Fundamentals</h4>
            <pre><code>{`import numpy as np

# Creating arrays
arr1d = np.array([1, 2, 3, 4, 5])                    # 1D array
arr2d = np.array([[1, 2, 3], [4, 5, 6]])            # 2D array
zeros = np.zeros((3, 4))                             # 3x4 array of zeros
ones = np.ones((2, 3))                               # 2x3 array of ones
identity = np.eye(3)                                 # 3x3 identity matrix
random_arr = np.random.randn(1000, 5)               # Random data (1000 samples, 5 features)

print(f"1D array: {arr1d}")
print(f"Shape: {arr1d.shape}, Dtype: {arr1d.dtype}")
print(f"2D array:\\n{arr2d}")
print(f"Random array shape: {random_arr.shape}")

# Array indexing and slicing
print(f"\\nFirst element: {arr1d[0]}")              # First element
print(f"Last element: {arr1d[-1]}")                 # Last element  
print(f"First row of 2D: {arr2d[0]}")               # First row
print(f"Element at (1,2): {arr2d[1, 2]}")           # Specific element
print(f"First 3 elements: {arr1d[:3]}")             # Slice
print(f"Every other element: {arr1d[::2]}")         # Step slicing

# Boolean indexing (very important for data filtering)
data = np.array([1, 5, 8, 3, 9, 2, 7])
mask = data > 5                                      # Boolean array
filtered = data[mask]                                # Elements > 5
print(f"\\nOriginal data: {data}")
print(f"Mask (>5): {mask}")
print(f"Filtered data: {filtered}")

# Mathematical operations (vectorized!)
arr = np.array([1, 2, 3, 4, 5])
squared = arr ** 2                                   # Element-wise squaring
sqrt_arr = np.sqrt(arr)                             # Square root
log_arr = np.log(arr)                               # Natural logarithm
exp_arr = np.exp(arr)                               # Exponential

print(f"\\nOriginal: {arr}")
print(f"Squared: {squared}")
print(f"Square root: {sqrt_arr}")

# Statistical operations
dataset = np.random.normal(100, 15, 1000)          # Normal distribution (mean=100, std=15)
print(f"\\nDataset statistics:")
print(f"Mean: {np.mean(dataset):.2f}")
print(f"Median: {np.median(dataset):.2f}")
print(f"Std deviation: {np.std(dataset):.2f}")
print(f"Min: {np.min(dataset):.2f}")
print(f"Max: {np.max(dataset):.2f}")
print(f"25th percentile: {np.percentile(dataset, 25):.2f}")

# Broadcasting - different shaped arrays
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

# Add 1D array to each row of 2D array
result = arr_2d + arr_1d                            # Broadcasting!
print(f"\\nBroadcasting example:")
print(f"2D array:\\n{arr_2d}")  
print(f"1D array: {arr_1d}")
print(f"Result:\\n{result}")

# Linear algebra operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

matrix_mult = A @ B                                  # Matrix multiplication
element_mult = A * B                                 # Element-wise multiplication
transpose = A.T                                      # Transpose
determinant = np.linalg.det(A)                      # Determinant
eigenvals, eigenvecs = np.linalg.eig(A)            # Eigendecomposition

print(f"\\nLinear algebra:")
print(f"A @ B:\\n{matrix_mult}")
print(f"Determinant of A: {determinant}")
print(f"Eigenvalues: {eigenvals}")

# Reshaping and axis operations
data_3d = np.random.randn(10, 5, 3)                # 10 samples, 5 features, 3 channels
flattened = data_3d.reshape(-1, 15)                # Flatten to 2D: 10 x 15
mean_over_samples = np.mean(data_3d, axis=0)       # Mean across samples (axis 0)
mean_over_features = np.mean(data_3d, axis=1)      # Mean across features (axis 1)

print(f"\\n3D data shape: {data_3d.shape}")
print(f"Flattened shape: {flattened.shape}")
print(f"Mean over samples shape: {mean_over_samples.shape}")

# Advanced indexing
arr = np.arange(20).reshape(4, 5)                  # 4x5 array
row_indices = [0, 2, 3]
col_indices = [1, 3, 4]
selected = arr[row_indices, col_indices]           # Fancy indexing

print(f"\\nOriginal array:\\n{arr}")
print(f"Selected elements: {selected}")            # Elements at (0,1), (2,3), (3,4)

# Memory efficiency demonstration
python_list = list(range(1000000))                 # Python list
numpy_array = np.arange(1000000)                   # NumPy array

# NumPy arrays use much less memory and are much faster for operations!
print(f"\\nMemory comparison (1M elements):")
print(f"Python list: ~{python_list.__sizeof__() + sum(x.__sizeof__() for x in python_list[:100]) * 10000} bytes")  
print(f"NumPy array: {numpy_array.nbytes} bytes")`}</code></pre>
          </div>

          <div className="performance-comparison">
            <h4>Performance Comparison: Python vs NumPy</h4>
            <pre><code>{`import time

# Performance comparison
size = 1000000
python_list = list(range(size))
numpy_array = np.arange(size)

# Addition operation
start = time.time()
python_result = [x + 1 for x in python_list]
python_time = time.time() - start

start = time.time() 
numpy_result = numpy_array + 1
numpy_time = time.time() - start

print(f"Python list time: {python_time:.4f} seconds")
print(f"NumPy array time: {numpy_time:.4f} seconds")
print(f"NumPy is {python_time/numpy_time:.1f}x faster!")`}</code></pre>
          </div>
        </section>

        <section className="pandas-section">
          <h2>🐼 Pandas - Data Analysis Library</h2>
          <p>Pandas provides high-performance, easy-to-use data structures and data analysis tools. It's built on top of NumPy and is essential for data manipulation.</p>

          <h3>Key Data Structures</h3>
          <ul>
            <li><strong>Series:</strong> 1D labeled array (like a column in Excel)</li>
            <li><strong>DataFrame:</strong> 2D labeled data structure (like a spreadsheet)</li>
          </ul>

          <div className="code-example">
            <h4>Pandas Fundamentals</h4>
            <pre><code>{`import pandas as pd
import numpy as np

# Creating a Series
temperatures = pd.Series([23.5, 25.1, 22.8, 26.2, 24.9], 
                        index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
print("Temperature Series:")
print(temperatures)
print(f"Wednesday temp: {temperatures['Wed']}")

# Creating a DataFrame from dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Salary': [50000, 60000, 70000, 55000, 65000],
    'Department': ['IT', 'HR', 'Finance', 'IT', 'Finance']
}

df = pd.DataFrame(data)
print(f"\\nDataFrame:")
print(df)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\\n{df.dtypes}")

# Basic DataFrame operations
print(f"\\nFirst 3 rows:")
print(df.head(3))
print(f"\\nLast 2 rows:")  
print(df.tail(2))
print(f"\\nDataFrame info:")
print(df.info())
print(f"\\nSummary statistics:")
print(df.describe())

# Selecting data
print(f"\\nSelect single column:")
print(df['Name'])
print(f"\\nSelect multiple columns:")
print(df[['Name', 'Salary']])
print(f"\\nSelect rows by condition:")
high_salary = df[df['Salary'] > 60000]
print(high_salary)

# Adding new columns
df['Bonus'] = df['Salary'] * 0.1                    # 10% bonus
df['Total_Compensation'] = df['Salary'] + df['Bonus']
df['Experience'] = np.random.randint(1, 10, len(df)) # Random experience

print(f"\\nDataFrame with new columns:")
print(df)

# Grouping and aggregation
dept_stats = df.groupby('Department').agg({
    'Salary': ['mean', 'min', 'max'],
    'Age': 'mean',
    'Experience': 'sum'
})
print(f"\\nGrouped statistics by department:")
print(dept_stats)

# Filtering and sorting
it_employees = df[df['Department'] == 'IT'].sort_values('Salary', ascending=False)
print(f"\\nIT employees sorted by salary:")
print(it_employees)

# Handling missing data
df_with_na = df.copy()
df_with_na.loc[1, 'Salary'] = np.nan              # Introduce missing value
df_with_na.loc[3, 'Age'] = np.nan

print(f"\\nDataFrame with missing values:")
print(df_with_na)
print(f"\\nMissing values count:")
print(df_with_na.isnull().sum())

# Fill missing values
df_filled = df_with_na.fillna({
    'Salary': df_with_na['Salary'].mean(),          # Fill with mean
    'Age': df_with_na['Age'].median()               # Fill with median
})
print(f"\\nAfter filling missing values:")
print(df_filled)

# String operations
df['Name_Length'] = df['Name'].str.len()
df['Name_Upper'] = df['Name'].str.upper()
df['Email'] = df['Name'].str.lower() + '@company.com'

print(f"\\nString operations:")
print(df[['Name', 'Name_Length', 'Email']])

# Date/time operations
dates = pd.date_range('2024-01-01', periods=len(df), freq='D')
df['Join_Date'] = dates
df['Days_Since_Join'] = (pd.Timestamp.now() - df['Join_Date']).dt.days

print(f"\\nDate operations:")
print(df[['Name', 'Join_Date', 'Days_Since_Join']])

# Pivot tables
pivot = df.pivot_table(values='Salary', 
                      index='Department', 
                      aggfunc=['mean', 'count'])
print(f"\\nPivot table:")
print(pivot)

# Merging DataFrames
performance_data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Performance_Score': [8.5, 7.2, 9.1]
})

merged_df = df.merge(performance_data, on='Name', how='left')
print(f"\\nMerged DataFrame:")
print(merged_df[['Name', 'Department', 'Salary', 'Performance_Score']])`}</code></pre>
          </div>

          <h3>Data Loading and Saving</h3>
          <div className="code-example">
            <pre><code>{`# Reading different file formats
# df = pd.read_csv('data.csv')                    # CSV files
# df = pd.read_excel('data.xlsx', sheet_name=0)   # Excel files  
# df = pd.read_json('data.json')                  # JSON files
# df = pd.read_sql('SELECT * FROM table', conn)   # SQL databases

# Saving data
# df.to_csv('output.csv', index=False)            # Save as CSV
# df.to_excel('output.xlsx', index=False)         # Save as Excel
# df.to_json('output.json', orient='records')     # Save as JSON

# Example: Creating sample CSV data
sample_data = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=100),
    'Sales': np.random.normal(1000, 200, 100),
    'Customers': np.random.poisson(50, 100),
    'Product': np.random.choice(['A', 'B', 'C'], 100)
})

# Save sample data (commented out to avoid file creation)
# sample_data.to_csv('sample_sales_data.csv', index=False)
print("Sample sales data:")
print(sample_data.head())`}</code></pre>
          </div>
        </section>

        <section className="visualization-section">
          <h2>📊 Data Visualization</h2>
          <p>Visualization is crucial for understanding data patterns, relationships, and distributions. Python offers powerful libraries for creating insightful visualizations.</p>

          <div className="code-example">
            <h4>Matplotlib - The Foundation</h4>
            <pre><code>{`import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create sample dataset
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
data = {
    'height': np.random.normal(170, 10, n_samples),
    'weight': np.random.normal(70, 15, n_samples),
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.lognormal(10, 0.5, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
}

# Add correlation between height and weight
data['weight'] = data['height'] * 0.8 + np.random.normal(0, 5, n_samples)
df = pd.DataFrame(data)

# 1. Line plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Time series example
dates = pd.date_range('2020-01-01', periods=365)
stock_price = 100 + np.cumsum(np.random.randn(365) * 0.5)
axes[0, 0].plot(dates, stock_price)
axes[0, 0].set_title('Stock Price Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price ($)')

# 2. Histogram
axes[0, 1].hist(df['height'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 1].set_title('Height Distribution')
axes[0, 1].set_xlabel('Height (cm)')
axes[0, 1].set_ylabel('Frequency')

# 3. Scatter plot
scatter = axes[0, 2].scatter(df['height'], df['weight'], 
                           c=df['age'], cmap='viridis', alpha=0.6)
axes[0, 2].set_title('Height vs Weight (colored by Age)')
axes[0, 2].set_xlabel('Height (cm)')
axes[0, 2].set_ylabel('Weight (kg)')
plt.colorbar(scatter, ax=axes[0, 2], label='Age')

# 4. Bar plot
education_counts = df['education'].value_counts()
axes[1, 0].bar(education_counts.index, education_counts.values, 
              color=['#FF9999', '#66B2FF', '#99FF99', '#FFD700'])
axes[1, 0].set_title('Education Distribution')
axes[1, 0].set_xlabel('Education Level')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5. Box plot
gender_groups = [df[df['gender'] == 'Male']['income'], 
                df[df['gender'] == 'Female']['income']]
axes[1, 1].boxplot(gender_groups, labels=['Male', 'Female'])
axes[1, 1].set_title('Income Distribution by Gender')
axes[1, 1].set_ylabel('Income ($)')

# 6. Pie chart
gender_counts = df['gender'].value_counts()
axes[1, 2].pie(gender_counts.values, labels=gender_counts.index, 
              autopct='%1.1f%%', startangle=90)
axes[1, 2].set_title('Gender Distribution')

plt.tight_layout()
# plt.show()  # Uncomment to display plots

print("Matplotlib plots created successfully!")

# Advanced plotting with Seaborn
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Correlation heatmap
numeric_cols = ['height', 'weight', 'age', 'income']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
           center=0, ax=axes[0, 0])
axes[0, 0].set_title('Correlation Heatmap')

# 2. Pairplot (would be separate figure in practice)
# sns.pairplot(df[numeric_cols])

# 3. Violin plot
sns.violinplot(data=df, x='education', y='income', ax=axes[0, 1])
axes[0, 1].set_title('Income Distribution by Education')
axes[0, 1].tick_params(axis='x', rotation=45)

# 4. Joint plot showing distribution and relationship
# Create joint plot data
x = df['height'].values
y = df['weight'].values
axes[1, 0].scatter(x, y, alpha=0.5)
axes[1, 0].set_xlabel('Height (cm)')
axes[1, 0].set_ylabel('Weight (kg)')
axes[1, 0].set_title('Height vs Weight')

# Add marginal histograms manually
from matplotlib.patches import Rectangle
# This is simplified - seaborn's jointplot is more sophisticated

# 5. Regression plot
sns.regplot(data=df, x='height', y='weight', ax=axes[1, 1])
axes[1, 1].set_title('Height vs Weight with Regression Line')

plt.tight_layout()
# plt.show()

print("Seaborn plots created successfully!")

# Interactive plotting with plotly (conceptual - would need plotly installed)
"""
import plotly.express as px
import plotly.graph_objects as go

# Interactive scatter plot
fig = px.scatter(df, x='height', y='weight', color='gender', 
                size='age', hover_data=['education', 'income'])
fig.update_layout(title='Interactive Height vs Weight Plot')
# fig.show()

# Interactive time series
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=stock_price, mode='lines', 
                        name='Stock Price'))
fig.update_layout(title='Interactive Stock Price Chart',
                 xaxis_title='Date', yaxis_title='Price ($)')
# fig.show()
"""

print("Data visualization examples completed!")

# Plotting best practices
plotting_tips = """
📊 Data Visualization Best Practices:

1. Choose the Right Chart Type:
   - Line plots: Time series, trends
   - Histograms: Distribution of single variable
   - Scatter plots: Relationships between variables
   - Bar plots: Categorical comparisons
   - Box plots: Distribution summaries
   - Heatmaps: Correlation matrices

2. Design Principles:
   - Clear, descriptive titles
   - Labeled axes with units
   - Appropriate color schemes
   - Proper scaling
   - Remove unnecessary clutter

3. Color Guidelines:
   - Use colorblind-friendly palettes
   - Consistent color meaning
   - Highlight important data points
   - Avoid too many colors

4. For Machine Learning:
   - Plot training/validation curves
   - Visualize feature distributions
   - Show model predictions vs actual
   - Create confusion matrices
   - Plot ROC curves and precision-recall curves
"""

print(plotting_tips)`}</code></pre>
          </div>
        </section>

        <section className="eda-section">
          <h2>🔍 Exploratory Data Analysis (EDA)</h2>
          <p>EDA is the process of investigating datasets to discover patterns, spot anomalies, and check assumptions using statistical summaries and visualizations.</p>

          <div className="code-example">
            <h4>Complete EDA Pipeline</h4>
            <pre><code>{`import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load and examine the data
def initial_data_exploration(df):
    """Perform initial data exploration"""
    print("=" * 50)
    print("INITIAL DATA EXPLORATION")
    print("=" * 50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\\nColumn names and types:")
    print(df.dtypes)
    
    print(f"\\nFirst few rows:")
    print(df.head())
    
    print(f"\\nSummary statistics:")
    print(df.describe(include='all'))
    
    print(f"\\nMissing values:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_percent})
    print(missing_df[missing_df['Count'] > 0])
    
    print(f"\\nDuplicate rows: {df.duplicated().sum()}")
    
    return df

def analyze_numerical_features(df, numerical_cols):
    """Analyze numerical features"""
    print("\\n" + "=" * 50) 
    print("NUMERICAL FEATURES ANALYSIS")
    print("=" * 50)
    
    for col in numerical_cols:
        print(f"\\nAnalyzing {col}:")
        
        # Basic statistics
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std: {df[col].std():.2f}")
        print(f"  Min: {df[col].min():.2f}")
        print(f"  Max: {df[col].max():.2f}")
        
        # Skewness and kurtosis
        skewness = stats.skew(df[col].dropna())
        kurtosis = stats.kurtosis(df[col].dropna())
        print(f"  Skewness: {skewness:.2f} ({'Right' if skewness > 0 else 'Left'} skewed)")
        print(f"  Kurtosis: {kurtosis:.2f}")
        
        # Outlier detection using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"  Outliers (IQR method): {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

def analyze_categorical_features(df, categorical_cols):
    """Analyze categorical features"""
    print("\\n" + "=" * 50)
    print("CATEGORICAL FEATURES ANALYSIS") 
    print("=" * 50)
    
    for col in categorical_cols:
        print(f"\\nAnalyzing {col}:")
        
        value_counts = df[col].value_counts()
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most frequent: {value_counts.index[0]} ({value_counts.iloc[0]} times)")
        print(f"  Value distribution:")
        
        for value, count in value_counts.head().items():
            percentage = (count / len(df)) * 100
            print(f"    {value}: {count} ({percentage:.1f}%)")

def correlation_analysis(df, numerical_cols):
    """Analyze correlations between numerical features"""
    print("\\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)
    
    corr_matrix = df[numerical_cols].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation threshold
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_val
                ))
    
    if high_corr_pairs:
        print("High correlations found:")
        for col1, col2, corr_val in high_corr_pairs:
            print(f"  {col1} - {col2}: {corr_val:.3f}")
    else:
        print("No high correlations (>0.7) found")
    
    return corr_matrix

def detect_anomalies(df, numerical_cols):
    """Detect anomalies using multiple methods"""
    print("\\n" + "=" * 50)
    print("ANOMALY DETECTION")
    print("=" * 50)
    
    anomaly_indices = set()
    
    for col in numerical_cols:
        # Z-score method
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        z_outliers = df[z_scores > 3].index
        
        # IQR method  
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        
        print(f"{col}:")
        print(f"  Z-score outliers: {len(z_outliers)}")
        print(f"  IQR outliers: {len(iqr_outliers)}")
        
        anomaly_indices.update(z_outliers)
        anomaly_indices.update(iqr_outliers)
    
    print(f"\\nTotal unique anomaly indices: {len(anomaly_indices)}")
    return list(anomaly_indices)

def create_eda_visualizations(df, numerical_cols, categorical_cols):
    """Create comprehensive EDA visualizations"""
    print("\\n" + "=" * 50)
    print("CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribution plots for numerical features
    n_num_cols = len(numerical_cols)
    if n_num_cols > 0:
        fig, axes = plt.subplots(nrows=(n_num_cols+2)//3, ncols=3, 
                               figsize=(15, 5*((n_num_cols+2)//3)))
        if n_num_cols == 1:
            axes = [axes]
        elif n_num_cols <= 3:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if n_num_cols > 1:
                ax = axes[i]
            else:
                ax = axes[0] if n_num_cols == 1 else axes
                
            # Histogram with KDE
            df[col].hist(bins=30, alpha=0.7, ax=ax, density=True)
            df[col].plot(kind='kde', ax=ax, color='red')
            ax.set_title(f'Distribution of {col}')
            ax.set_ylabel('Density')
        
        # Hide empty subplots
        if n_num_cols > 1:
            for i in range(n_num_cols, len(axes)):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        # plt.show()
    
    # 2. Correlation heatmap
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, square=True)
        plt.title('Correlation Heatmap')
        # plt.show()
    
    # 3. Box plots for numerical features
    if n_num_cols > 0:
        fig, axes = plt.subplots(nrows=(n_num_cols+2)//3, ncols=3, 
                               figsize=(15, 5*((n_num_cols+2)//3)))
        if n_num_cols == 1:
            axes = [axes]
        elif n_num_cols <= 3:
            axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if n_num_cols > 1:
                ax = axes[i]
            else:
                ax = axes[0] if n_num_cols == 1 else axes
                
            df.boxplot(column=col, ax=ax)
            ax.set_title(f'Box Plot of {col}')
        
        # Hide empty subplots
        if n_num_cols > 1:
            for i in range(n_num_cols, len(axes)):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        # plt.show()
    
    print("Visualizations created successfully!")

# Example usage with sample data
def run_complete_eda():
    """Run complete EDA on sample dataset"""
    
    # Create sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'performance_score': np.random.uniform(1, 10, n_samples)
    })
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[outlier_indices, 'income'] *= 10
    
    # Define column types
    numerical_cols = ['age', 'income', 'education_years', 'experience', 'performance_score']
    categorical_cols = ['department', 'gender']
    
    # Run EDA pipeline
    df = initial_data_exploration(df)
    analyze_numerical_features(df, numerical_cols)
    analyze_categorical_features(df, categorical_cols)
    correlation_matrix = correlation_analysis(df, numerical_cols)
    anomalies = detect_anomalies(df, numerical_cols)
    create_eda_visualizations(df, numerical_cols, categorical_cols)
    
    return df

# Run the complete EDA
sample_df = run_complete_eda()

print("\\n" + "=" * 50)
print("EDA COMPLETE!")
print("=" * 50)
print("\\nNext steps in your data science pipeline:")
print("1. Data cleaning and preprocessing")
print("2. Feature engineering")
print("3. Feature selection")
print("4. Model training and evaluation")
print("5. Model deployment")`}</code></pre>
          </div>
        </section>

        <section className="preprocessing-section">
          <h2>🔧 Data Preprocessing</h2>
          <p>Data preprocessing is crucial for preparing raw data for machine learning models. This includes cleaning, transforming, and scaling data.</p>

          <div className="code-example">
            <h4>Complete Preprocessing Pipeline</h4>
            <pre><code>{`import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                 LabelEncoder, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class DataPreprocessor:
    """Complete data preprocessing pipeline"""
    
    def __init__(self):
        self.numerical_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
        self.target_encoder = None
        
    def identify_column_types(self, df, target_column=None):
        """Automatically identify numerical and categorical columns"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column from features if specified
        if target_column:
            if target_column in numerical_cols:
                numerical_cols.remove(target_column)
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
        
        return numerical_cols, categorical_cols
    
    def handle_missing_values(self, df, strategy='mean'):
        """Handle missing values in dataset"""
        print("Handling missing values...")
        
        missing_info = df.isnull().sum()
        columns_with_missing = missing_info[missing_info > 0]
        
        if len(columns_with_missing) > 0:
            print(f"Found missing values in {len(columns_with_missing)} columns:")
            for col, count in columns_with_missing.items():
                percentage = (count / len(df)) * 100
                print(f"  {col}: {count} ({percentage:.1f}%)")
        
        return df
    
    def detect_and_handle_outliers(self, df, numerical_cols, method='iqr'):
        """Detect and handle outliers"""
        print(f"\\nDetecting outliers using {method} method...")
        
        df_cleaned = df.copy()
        total_outliers = 0
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    print(f"  {col}: {outliers_count} outliers")
                    # Cap outliers to bounds
                    df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
                    total_outliers += outliers_count
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_mask = z_scores > 3
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    print(f"  {col}: {outliers_count} outliers")
                    # Replace with median
                    median_val = df[col].median()
                    df_cleaned.loc[outliers_mask, col] = median_val
                    total_outliers += outliers_count
        
        print(f"Total outliers handled: {total_outliers}")
        return df_cleaned
    
    def create_preprocessing_pipeline(self, numerical_cols, categorical_cols, 
                                   scaler_type='standard'):
        """Create preprocessing pipeline"""
        print(f"\\nCreating preprocessing pipeline...")
        print(f"  Numerical columns: {len(numerical_cols)}")
        print(f"  Categorical columns: {len(categorical_cols)}")
        
        # Numerical preprocessing
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        
        # Categorical preprocessing
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])
        
        return self.preprocessor
    
    def fit_transform(self, X_train, y_train=None):
        """Fit preprocessor on training data and transform"""
        print("\\nFitting preprocessor on training data...")
        
        # Fit and transform features
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Handle target variable if provided
        if y_train is not None:
            if y_train.dtype == 'object':
                self.target_encoder = LabelEncoder()
                y_train_processed = self.target_encoder.fit_transform(y_train)
            else:
                y_train_processed = y_train
            
            return X_train_processed, y_train_processed
        
        return X_train_processed
    
    def transform(self, X_test, y_test=None):
        """Transform test data using fitted preprocessor"""
        print("Transforming test data...")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Transform features
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Handle target variable if provided
        if y_test is not None:
            if self.target_encoder is not None:
                y_test_processed = self.target_encoder.transform(y_test)
            else:
                y_test_processed = y_test
            
            return X_test_processed, y_test_processed
        
        return X_test_processed
    
    def get_feature_names(self, numerical_cols, categorical_cols):
        """Get feature names after preprocessing"""
        # Get categorical feature names after one-hot encoding
        if hasattr(self.preprocessor.named_transformers_['cat'], 'named_steps'):
            onehot_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
        else:
            cat_feature_names = []
        
        # Combine all feature names
        all_feature_names = list(numerical_cols) + list(cat_feature_names)
        return all_feature_names

# Example usage
def preprocessing_example():
    """Complete preprocessing example"""
    print("=" * 60)
    print("DATA PREPROCESSING EXAMPLE")
    print("=" * 60)
    
    # Create sample dataset with various data quality issues
    np.random.seed(42)
    n_samples = 1000
    
    # Generate base data
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience': np.random.randint(0, 40, n_samples),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_samples),
        'performance_score': np.random.uniform(1, 10, n_samples),
        'target': np.random.choice(['Low', 'Medium', 'High'], n_samples)
    })
    
    # Introduce data quality issues
    # 1. Missing values
    missing_indices = np.random.choice(df.index, size=100, replace=False)
    df.loc[missing_indices[:50], 'income'] = np.nan
    df.loc[missing_indices[50:], 'department'] = np.nan
    
    # 2. Outliers
    outlier_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[outlier_indices, 'income'] *= 10
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Missing values:\\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Identify column types
    target_col = 'target'
    numerical_cols, categorical_cols = preprocessor.identify_column_types(df, target_col)
    
    print(f"\\nNumerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Handle outliers
    df_cleaned = preprocessor.detect_and_handle_outliers(df, numerical_cols, method='iqr')
    
    # Split data
    X = df_cleaned.drop(target_col, axis=1)
    y = df_cleaned[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\\nTrain set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Create and fit preprocessing pipeline
    pipeline = preprocessor.create_preprocessing_pipeline(
        numerical_cols, categorical_cols, scaler_type='standard'
    )
    
    # Fit and transform training data
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    
    print(f"\\nProcessed train shape: {X_train_processed.shape}")
    print(f"Processed test shape: {X_test_processed.shape}")
    
    # Get feature names
    feature_names = preprocessor.get_feature_names(numerical_cols, categorical_cols)
    print(f"\\nTotal features after preprocessing: {len(feature_names)}")
    print(f"Feature names: {feature_names[:10]}...")  # Show first 10
    
    # Show preprocessing effects
    print("\\n" + "=" * 60)
    print("PREPROCESSING EFFECTS")
    print("=" * 60)
    
    print("\\nOriginal data (first few numerical columns):")
    print(X_train[numerical_cols[:3]].describe())
    
    print("\\nProcessed data (same columns, now standardized):")
    processed_df = pd.DataFrame(X_train_processed[:, :3], 
                              columns=numerical_cols[:3])
    print(processed_df.describe())
    
    print("\\nTarget encoding:")
    if hasattr(preprocessor, 'target_encoder') and preprocessor.target_encoder:
        print(f"Original target values: {sorted(y_train.unique())}")
        print(f"Encoded target values: {sorted(np.unique(y_train_processed))}")
        print(f"Encoding mapping: {dict(zip(preprocessor.target_encoder.classes_, 
                                           range(len(preprocessor.target_encoder.classes_))))}")
    
    return X_train_processed, X_test_processed, y_train_processed, y_test_processed

# Run the preprocessing example
X_train_proc, X_test_proc, y_train_proc, y_test_proc = preprocessing_example()

print("\\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)
print("\\nData is now ready for machine learning models!")
print("Next steps:")
print("1. Feature selection/engineering")
print("2. Model selection and training")
print("3. Model evaluation and tuning")
print("4. Model deployment")`}</code></pre>
          </div>
        </section>
      </div>
    </div>
  )
}

export default DataScience