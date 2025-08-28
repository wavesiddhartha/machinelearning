import { useState } from 'react'

function MachineLearningComplete() {
  const [activeSection, setActiveSection] = useState('introduction')
  const [expandedCode, setExpandedCode] = useState(null)

  const sections = [
    { id: 'introduction', title: 'What is Machine Learning?', icon: '🤖' },
    { id: 'mathematics', title: 'ML Mathematics Foundation', icon: '📊' },
    { id: 'data-preprocessing', title: 'Data Preprocessing', icon: '🧹' },
    { id: 'supervised-learning', title: 'Supervised Learning', icon: '👨‍🏫' },
    { id: 'unsupervised-learning', title: 'Unsupervised Learning', icon: '🔍' },
    { id: 'neural-networks', title: 'Neural Networks & Deep Learning', icon: '🧠' },
    { id: 'model-evaluation', title: 'Model Evaluation & Validation', icon: '📏' },
    { id: 'practical-implementation', title: 'Practical Implementation', icon: '💻' },
    { id: 'advanced-topics', title: 'Advanced ML Topics', icon: '🚀' },
    { id: 'real-world-projects', title: 'Real-World Projects', icon: '🌍' }
  ]

  const codeExamples = {
    basicML: `# Complete Machine Learning Example: House Price Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Create sample dataset
np.random.seed(42)
n_samples = 1000

# Generate features
house_size = np.random.normal(2000, 500, n_samples)  # Square feet
bedrooms = np.random.poisson(3, n_samples)           # Number of bedrooms
age = np.random.uniform(0, 50, n_samples)            # Age of house
location_score = np.random.uniform(1, 10, n_samples) # Location desirability

# Create target variable (price) with realistic relationships
price = (
    house_size * 150 +           # $150 per sq ft
    bedrooms * 10000 +           # $10k per bedroom
    (50 - age) * 1000 +         # Newer houses cost more
    location_score * 5000 +      # Location premium
    np.random.normal(0, 20000, n_samples)  # Random noise
)

# Create DataFrame
data = pd.DataFrame({
    'house_size': house_size,
    'bedrooms': bedrooms,
    'age': age,
    'location_score': location_score,
    'price': price
})

print("Dataset Info:")
print(data.describe())
print("\\nFirst few rows:")
print(data.head())

# Split features and target
X = data[['house_size', 'bedrooms', 'age', 'location_score']]
y = data['price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\\nModel Performance:")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\\nFeature Importance:")
print(feature_importance)

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

# Make a prediction for a new house
new_house = pd.DataFrame({
    'house_size': [2200],
    'bedrooms': [4],
    'age': [10],
    'location_score': [8.5]
})

new_house_scaled = scaler.transform(new_house)
predicted_price = model.predict(new_house_scaled)[0]

print(f"\\nPredicted price for new house: {predicted_price:,.2f}")`,

    dataPreprocessing: `# Comprehensive Data Preprocessing Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load and create sample messy dataset
np.random.seed(42)
n_samples = 1000

# Create dataset with common data problems
data = pd.DataFrame({
    'age': np.random.normal(35, 12, n_samples),
    'income': np.random.lognormal(10, 1, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'experience': np.random.exponential(5, n_samples),
    'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle'], n_samples),
    'performance_score': np.random.uniform(0, 100, n_samples)
})

# Introduce missing values (realistic scenario)
missing_mask_age = np.random.random(n_samples) < 0.05  # 5% missing
missing_mask_income = np.random.random(n_samples) < 0.08  # 8% missing
missing_mask_education = np.random.random(n_samples) < 0.03  # 3% missing

data.loc[missing_mask_age, 'age'] = np.nan
data.loc[missing_mask_income, 'income'] = np.nan
data.loc[missing_mask_education, 'education'] = np.nan

# Introduce outliers
outlier_indices = np.random.choice(n_samples, size=20, replace=False)
data.loc[outlier_indices, 'income'] *= 10  # Extreme incomes

print("=== ORIGINAL DATA ANALYSIS ===")
print("Dataset shape:", data.shape)
print("\\nData types:")
print(data.dtypes)
print("\\nMissing values:")
print(data.isnull().sum())
print("\\nBasic statistics:")
print(data.describe())

# 1. EXPLORATORY DATA ANALYSIS
def perform_eda(df):
    """Perform exploratory data analysis"""
    
    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(numerical_cols[:4]):
        row, col_idx = i // 2, i % 2
        if i < 4:
            axes[row, col_idx].hist(df[col].dropna(), bins=30, alpha=0.7)
            axes[row, col_idx].set_title(f'Distribution of {col}')
            axes[row, col_idx].set_xlabel(col)
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for i, col in enumerate(categorical_cols[:2]):
        if i < 2:
            axes[i//2, 2].pie(df[col].value_counts().values, 
                             labels=df[col].value_counts().index,
                             autopct='%1.1f%%')
            axes[i//2, 2].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation matrix for numerical features
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

perform_eda(data)

# 2. HANDLING MISSING VALUES
class MissingValueHandler:
    """Handle different types of missing values"""
    
    def __init__(self, strategy='auto'):
        self.strategy = strategy
        self.imputers = {}
    
    def fit_transform(self, df):
        df_copy = df.copy()
        
        # Numerical columns: use median (robust to outliers)
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
        numerical_imputer = SimpleImputer(strategy='median')
        
        for col in numerical_cols:
            if df_copy[col].isnull().any():
                df_copy[[col]] = numerical_imputer.fit_transform(df_copy[[col]])
                self.imputers[col] = numerical_imputer
                print(f"Imputed {col} missing values with median: {df_copy[col].median():.2f}")
        
        # Categorical columns: use mode (most frequent)
        categorical_cols = df_copy.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df_copy[col].isnull().any():
                mode_value = df_copy[col].mode()[0]
                df_copy[col].fillna(mode_value, inplace=True)
                print(f"Imputed {col} missing values with mode: {mode_value}")
        
        return df_copy

# 3. OUTLIER DETECTION AND HANDLING
class OutlierHandler:
    """Detect and handle outliers using various methods"""
    
    def __init__(self, method='iqr', threshold=1.5):
        self.method = method
        self.threshold = threshold
        self.bounds = {}
    
    def detect_outliers_iqr(self, series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def detect_outliers_zscore(self, series, threshold=3):
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    def handle_outliers(self, df, columns=None):
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if self.method == 'iqr':
                outliers = self.detect_outliers_iqr(df_copy[col])
            elif self.method == 'zscore':
                outliers = self.detect_outliers_zscore(df_copy[col])
            
            n_outliers = outliers.sum()
            print(f"Found {n_outliers} outliers in {col}")
            
            if n_outliers > 0:
                # Cap outliers at percentiles
                lower_cap = df_copy[col].quantile(0.05)
                upper_cap = df_copy[col].quantile(0.95)
                
                df_copy.loc[df_copy[col] < lower_cap, col] = lower_cap
                df_copy.loc[df_copy[col] > upper_cap, col] = upper_cap
                
                self.bounds[col] = (lower_cap, upper_cap)
                print(f"Capped {col} outliers to range [{lower_cap:.2f}, {upper_cap:.2f}]")
        
        return df_copy

# 4. FEATURE ENGINEERING
class FeatureEngineer:
    """Create new features from existing ones"""
    
    def __init__(self):
        self.encoders = {}
    
    def create_features(self, df):
        df_copy = df.copy()
        
        # Numerical feature engineering
        df_copy['income_per_year_experience'] = df_copy['income'] / (df_copy['experience'] + 1)
        df_copy['age_group'] = pd.cut(df_copy['age'], 
                                    bins=[0, 25, 35, 50, 100], 
                                    labels=['Young', 'Adult', 'Middle-aged', 'Senior'])
        
        # Log transformation for skewed features
        df_copy['log_income'] = np.log1p(df_copy['income'])
        
        # Interaction features
        df_copy['education_experience_interaction'] = (
            df_copy['education'].map({'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}) * 
            df_copy['experience']
        )
        
        print("Created new features:")
        print("- income_per_year_experience")
        print("- age_group")
        print("- log_income") 
        print("- education_experience_interaction")
        
        return df_copy

# 5. ENCODING CATEGORICAL VARIABLES
class CategoricalEncoder:
    """Handle categorical variable encoding"""
    
    def __init__(self):
        self.label_encoders = {}
        self.onehot_encoder = None
    
    def encode_features(self, df):
        df_copy = df.copy()
        
        # Ordinal encoding for education (has natural order)
        education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
        df_copy['education_encoded'] = df_copy['education'].map(education_mapping)
        
        # One-hot encoding for city (no natural order)
        city_dummies = pd.get_dummies(df_copy['city'], prefix='city')
        df_copy = pd.concat([df_copy, city_dummies], axis=1)
        
        # Age group encoding
        if 'age_group' in df_copy.columns:
            age_group_dummies = pd.get_dummies(df_copy['age_group'], prefix='age_group')
            df_copy = pd.concat([df_copy, age_group_dummies], axis=1)
        
        print("Encoded categorical variables:")
        print("- education: ordinal encoding")
        print("- city: one-hot encoding")
        print("- age_group: one-hot encoding")
        
        return df_copy

# 6. FEATURE SCALING
class FeatureScaler:
    """Scale numerical features"""
    
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = None
    
    def scale_features(self, df, exclude_cols=None):
        df_copy = df.copy()
        
        if exclude_cols is None:
            exclude_cols = []
        
        # Select numerical columns for scaling
        numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
        
        df_copy[cols_to_scale] = self.scaler.fit_transform(df_copy[cols_to_scale])
        
        print(f"Scaled {len(cols_to_scale)} numerical features using {self.method} scaling")
        
        return df_copy

# Execute the complete preprocessing pipeline
print("\\n=== PREPROCESSING PIPELINE ===")

# Step 1: Handle missing values
missing_handler = MissingValueHandler()
data_clean = missing_handler.fit_transform(data)

# Step 2: Handle outliers
outlier_handler = OutlierHandler(method='iqr', threshold=2.0)
data_clean = outlier_handler.handle_outliers(data_clean, ['income', 'age'])

# Step 3: Feature engineering
feature_engineer = FeatureEngineer()
data_engineered = feature_engineer.create_features(data_clean)

# Step 4: Encode categorical variables
encoder = CategoricalEncoder()
data_encoded = encoder.encode_features(data_engineered)

# Step 5: Feature scaling
scaler = FeatureScaler()
data_final = scaler.scale_features(data_encoded, 
                                 exclude_cols=['performance_score'])  # Keep target unscaled

print("\\n=== FINAL DATASET INFO ===")
print("Shape:", data_final.shape)
print("\\nColumns:", list(data_final.columns))
print("\\nMissing values:", data_final.isnull().sum().sum())
print("\\nFinal dataset ready for machine learning!")`,

    supervisedLearning: `# Complete Guide to Supervised Learning Algorithms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score

# Import various algorithms
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

# Create comprehensive datasets for demonstration

# CLASSIFICATION DATASET: Customer Purchase Prediction
np.random.seed(42)
n_customers = 1000

# Generate customer features
age = np.random.normal(40, 15, n_customers)
income = np.random.lognormal(10, 0.5, n_customers)
spending_score = np.random.uniform(0, 100, n_customers)
previous_purchases = np.random.poisson(5, n_customers)
time_on_site = np.random.exponential(10, n_customers)

# Create target: will customer make a purchase?
# Higher chance for: younger customers, higher income, higher spending score
purchase_probability = (
    0.3 +
    0.2 * (50 - age) / 50 +  # Younger = higher probability
    0.3 * (income - income.min()) / (income.max() - income.min()) +
    0.2 * spending_score / 100 +
    0.1 * np.minimum(previous_purchases / 10, 1) +
    0.1 * np.minimum(time_on_site / 20, 1)
)

will_purchase = np.random.binomial(1, purchase_probability)

# Create DataFrame
classification_data = pd.DataFrame({
    'age': age,
    'income': income,
    'spending_score': spending_score,
    'previous_purchases': previous_purchases,
    'time_on_site': time_on_site,
    'will_purchase': will_purchase
})

print("=== CLASSIFICATION PROBLEM: CUSTOMER PURCHASE PREDICTION ===")
print("Dataset shape:", classification_data.shape)
print("Target distribution:")
print(classification_data['will_purchase'].value_counts(normalize=True))
print()

# REGRESSION DATASET: House Price Prediction (already created in previous example)
# We'll use the same house price dataset

regression_data = pd.DataFrame({
    'house_size': np.random.normal(2000, 500, n_customers),
    'bedrooms': np.random.poisson(3, n_customers),
    'age': np.random.uniform(0, 50, n_customers),
    'location_score': np.random.uniform(1, 10, n_customers)
})

regression_data['price'] = (
    regression_data['house_size'] * 150 +
    regression_data['bedrooms'] * 10000 +
    (50 - regression_data['age']) * 1000 +
    regression_data['location_score'] * 5000 +
    np.random.normal(0, 20000, n_customers)
)

print("=== REGRESSION PROBLEM: HOUSE PRICE PREDICTION ===")
print("Dataset shape:", regression_data.shape)
print("Target statistics:")
print(regression_data['price'].describe())
print()

class SupervisedLearningDemo:
    """Comprehensive demonstration of supervised learning algorithms"""
    
    def __init__(self):
        self.classification_results = {}
        self.regression_results = {}
    
    def prepare_data(self, data, target_col):
        """Prepare data for machine learning"""
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if target_col == 'will_purchase' else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
    
    def evaluate_classification(self, model, X_test, y_test, model_name):
        """Evaluate classification model"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\\n--- {model_name} Results ---")
        print(f"Accuracy: {accuracy:.3f}")
        print("\\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        return accuracy
    
    def evaluate_regression(self, model, X_test, y_test, model_name):
        """Evaluate regression model"""
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"\\n--- {model_name} Results ---")
        print(f"R² Score: {r2:.3f}")
        print(f"RMSE: {rmse:,.2f}")
        print(f"MSE: {mse:,.2f}")
        
        # Prediction vs Actual plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{model_name} - Predictions vs Actual')
        plt.show()
        
        return r2
    
    def run_classification_algorithms(self, data):
        """Run all classification algorithms"""
        print("\\n" + "="*50)
        print("CLASSIFICATION ALGORITHMS COMPARISON")
        print("="*50)
        
        # Prepare data
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = \\
            self.prepare_data(data, 'will_purchase')
        
        # 1. LOGISTIC REGRESSION
        print("\\n1. LOGISTIC REGRESSION")
        print("="*30)
        print("Best for: Linear relationships, interpretability, probability estimates")
        print("Pros: Fast, interpretable, provides probabilities")
        print("Cons: Assumes linear relationship, sensitive to outliers")
        
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train_scaled, y_train)
        acc_lr = self.evaluate_classification(log_reg, X_test_scaled, y_test, "Logistic Regression")
        
        # Feature importance for logistic regression
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': log_reg.coef_[0],
            'abs_coefficient': np.abs(log_reg.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        print("\\nFeature Importance (Coefficients):")
        print(feature_importance)
        
        # 2. DECISION TREE
        print("\\n2. DECISION TREE")
        print("="*30)
        print("Best for: Non-linear relationships, interpretability, mixed data types")
        print("Pros: Interpretable, handles non-linear relationships, no scaling needed")
        print("Cons: Prone to overfitting, unstable")
        
        dt = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt.fit(X_train, y_train)
        acc_dt = self.evaluate_classification(dt, X_test, y_test, "Decision Tree")
        
        # Feature importance for decision tree
        feature_importance_dt = pd.DataFrame({
            'feature': X_train.columns,
            'importance': dt.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\\nFeature Importance:")
        print(feature_importance_dt)
        
        # 3. RANDOM FOREST
        print("\\n3. RANDOM FOREST")
        print("="*30)
        print("Best for: High accuracy, handling overfitting, feature selection")
        print("Pros: Reduces overfitting, handles missing values, feature importance")
        print("Cons: Less interpretable, can overfit with many trees")
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        acc_rf = self.evaluate_classification(rf, X_test, y_test, "Random Forest")
        
        # 4. SUPPORT VECTOR MACHINE
        print("\\n4. SUPPORT VECTOR MACHINE (SVM)")
        print("="*30)
        print("Best for: High-dimensional data, complex boundaries")
        print("Pros: Effective in high dimensions, memory efficient")
        print("Cons: Slow on large datasets, requires scaling")
        
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train_scaled, y_train)
        acc_svm = self.evaluate_classification(svm, X_test_scaled, y_test, "SVM")
        
        # 5. K-NEAREST NEIGHBORS
        print("\\n5. K-NEAREST NEIGHBORS (KNN)")
        print("="*30)
        print("Best for: Simple implementation, non-parametric")
        print("Pros: Simple, works well with small datasets")
        print("Cons: Computationally expensive, sensitive to irrelevant features")
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        acc_knn = self.evaluate_classification(knn, X_test_scaled, y_test, "KNN")
        
        # 6. NAIVE BAYES
        print("\\n6. NAIVE BAYES")
        print("="*30)
        print("Best for: Text classification, small datasets, fast prediction")
        print("Pros: Fast, works well with small data, handles multiple classes")
        print("Cons: Strong independence assumption")
        
        nb = GaussianNB()
        nb.fit(X_train_scaled, y_train)
        acc_nb = self.evaluate_classification(nb, X_test_scaled, y_test, "Naive Bayes")
        
        # 7. GRADIENT BOOSTING
        print("\\n7. GRADIENT BOOSTING")
        print("="*30)
        print("Best for: High accuracy, competitions, complex patterns")
        print("Pros: Usually highest accuracy, handles mixed data types")
        print("Cons: Prone to overfitting, requires tuning, slow training")
        
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train, y_train)
        acc_gb = self.evaluate_classification(gb, X_test, y_test, "Gradient Boosting")
        
        # Summary comparison
        self.classification_results = {
            'Logistic Regression': acc_lr,
            'Decision Tree': acc_dt,
            'Random Forest': acc_rf,
            'SVM': acc_svm,
            'KNN': acc_knn,
            'Naive Bayes': acc_nb,
            'Gradient Boosting': acc_gb
        }
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        algorithms = list(self.classification_results.keys())
        accuracies = list(self.classification_results.values())
        
        bars = plt.bar(algorithms, accuracies, color='skyblue', alpha=0.8)
        plt.title('Classification Algorithms Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\nBest performing algorithm: {max(self.classification_results, key=self.classification_results.get)}")
        print(f"Best accuracy: {max(self.classification_results.values()):.3f}")
    
    def run_regression_algorithms(self, data):
        """Run all regression algorithms"""
        print("\\n" + "="*50)
        print("REGRESSION ALGORITHMS COMPARISON")
        print("="*50)
        
        # Prepare data
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = \\
            self.prepare_data(data, 'price')
        
        # 1. LINEAR REGRESSION
        print("\\n1. LINEAR REGRESSION")
        print("="*30)
        print("Best for: Understanding relationships, baseline model, interpretability")
        print("Pros: Simple, interpretable, fast, provides confidence intervals")
        print("Cons: Assumes linear relationship, sensitive to outliers")
        
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        r2_lr = self.evaluate_regression(lr, X_test_scaled, y_test, "Linear Regression")
        
        # 2. RIDGE REGRESSION
        print("\\n2. RIDGE REGRESSION (L2 Regularization)")
        print("="*30)
        print("Best for: Multicollinearity, preventing overfitting")
        print("Pros: Handles multicollinearity, reduces overfitting")
        print("Cons: Doesn't perform feature selection")
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        r2_ridge = self.evaluate_regression(ridge, X_test_scaled, y_test, "Ridge Regression")
        
        # 3. LASSO REGRESSION
        print("\\n3. LASSO REGRESSION (L1 Regularization)")
        print("="*30)
        print("Best for: Feature selection, sparse models")
        print("Pros: Automatic feature selection, interpretable")
        print("Cons: Can select arbitrary features from correlated groups")
        
        lasso = Lasso(alpha=1.0)
        lasso.fit(X_train_scaled, y_train)
        r2_lasso = self.evaluate_regression(lasso, X_test_scaled, y_test, "Lasso Regression")
        
        # 4. DECISION TREE REGRESSION
        print("\\n4. DECISION TREE REGRESSION")
        print("="*30)
        print("Best for: Non-linear relationships, interpretability")
        print("Pros: Handles non-linear relationships, interpretable")
        print("Cons: Prone to overfitting, unstable")
        
        dt_reg = DecisionTreeRegressor(random_state=42, max_depth=8)
        dt_reg.fit(X_train, y_train)
        r2_dt = self.evaluate_regression(dt_reg, X_test, y_test, "Decision Tree Regression")
        
        # 5. RANDOM FOREST REGRESSION
        print("\\n5. RANDOM FOREST REGRESSION")
        print("="*30)
        print("Best for: High accuracy, robust predictions")
        print("Pros: Reduces overfitting, handles non-linearity, feature importance")
        print("Cons: Less interpretable, can overfit")
        
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, y_train)
        r2_rf = self.evaluate_regression(rf_reg, X_test, y_test, "Random Forest Regression")
        
        # 6. SUPPORT VECTOR REGRESSION
        print("\\n6. SUPPORT VECTOR REGRESSION (SVR)")
        print("="*30)
        print("Best for: High-dimensional data, robust to outliers")
        print("Pros: Effective in high dimensions, robust to outliers")
        print("Cons: Slow on large datasets, requires parameter tuning")
        
        svr = SVR(kernel='rbf', C=100, gamma=0.1)
        svr.fit(X_train_scaled, y_train)
        r2_svr = self.evaluate_regression(svr, X_test_scaled, y_test, "SVR")
        
        # 7. K-NEAREST NEIGHBORS REGRESSION
        print("\\n7. K-NEAREST NEIGHBORS REGRESSION")
        print("="*30)
        print("Best for: Local patterns, non-parametric")
        print("Pros: Simple, non-parametric, works well locally")
        print("Cons: Sensitive to irrelevant features, computationally expensive")
        
        knn_reg = KNeighborsRegressor(n_neighbors=5)
        knn_reg.fit(X_train_scaled, y_train)
        r2_knn = self.evaluate_regression(knn_reg, X_test_scaled, y_test, "KNN Regression")
        
        # Summary comparison
        self.regression_results = {
            'Linear Regression': r2_lr,
            'Ridge Regression': r2_ridge,
            'Lasso Regression': r2_lasso,
            'Decision Tree': r2_dt,
            'Random Forest': r2_rf,
            'SVR': r2_svr,
            'KNN': r2_knn
        }
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        algorithms = list(self.regression_results.keys())
        r2_scores = list(self.regression_results.values())
        
        bars = plt.bar(algorithms, r2_scores, color='lightgreen', alpha=0.8)
        plt.title('Regression Algorithms Comparison')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, r2 in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\\nBest performing algorithm: {max(self.regression_results, key=self.regression_results.get)}")
        print(f"Best R² score: {max(self.regression_results.values()):.3f}")

# Run the complete supervised learning demonstration
demo = SupervisedLearningDemo()

# Run classification algorithms
demo.run_classification_algorithms(classification_data)

# Run regression algorithms  
demo.run_regression_algorithms(regression_data)

print("\\n" + "="*60)
print("ALGORITHM SELECTION GUIDE")
print("="*60)
print("\\nCLASSIFICATION:")
print("• Logistic Regression: When you need interpretable results and probability estimates")
print("• Decision Tree: When you need full interpretability and have mixed data types")
print("• Random Forest: When you want high accuracy and feature importance")
print("• SVM: When you have high-dimensional data and complex decision boundaries")
print("• KNN: When you have small datasets and local patterns are important")
print("• Naive Bayes: For text classification and when features are independent")
print("• Gradient Boosting: When you need the highest accuracy and have time for tuning")

print("\\nREGRESSION:")
print("• Linear Regression: For understanding relationships and when linearity holds")
print("• Ridge: When you have multicollinearity issues")
print("• Lasso: When you need automatic feature selection")
print("• Decision Tree: When relationships are non-linear and you need interpretability")
print("• Random Forest: When you want robust, accurate predictions")
print("• SVR: For high-dimensional data with complex patterns")
print("• KNN: When local patterns are more important than global trends")

print("\\nGENERAL WORKFLOW:")
print("1. Start with simple models (Linear/Logistic Regression)")
print("2. Try ensemble methods (Random Forest, Gradient Boosting)")
print("3. Experiment with complex models (SVM, Neural Networks)")
print("4. Use cross-validation to compare models")
print("5. Tune hyperparameters for the best performing models")
print("6. Validate on unseen test data")`,

    neuralNetworks: `# Complete Guide to Neural Networks and Deep Learning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class NeuralNetworkGuide:
    """Comprehensive guide to neural networks and deep learning"""
    
    def __init__(self):
        self.models = {}
        
    def create_sample_data(self):
        """Create sample datasets for demonstration"""
        
        # Classification dataset
        X_class, y_class = make_classification(
            n_samples=2000, n_features=20, n_informative=15, 
            n_redundant=5, n_clusters_per_class=1, random_state=42
        )
        
        # Regression dataset  
        X_reg, y_reg = make_regression(
            n_samples=2000, n_features=15, noise=0.1, random_state=42
        )
        
        return X_class, y_class, X_reg, y_reg
    
    def build_basic_neural_network(self):
        """Build and explain a basic neural network from scratch"""
        
        print("=== BUILDING A NEURAL NETWORK FROM SCRATCH ===")
        print()
        
        class SimpleNeuralNetwork:
            def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
                # Initialize weights randomly
                self.W1 = np.random.randn(input_size, hidden_size) * 0.1
                self.b1 = np.zeros((1, hidden_size))
                self.W2 = np.random.randn(hidden_size, output_size) * 0.1
                self.b2 = np.zeros((1, output_size))
                self.learning_rate = learning_rate
                
                # Store for visualization
                self.costs = []
            
            def sigmoid(self, z):
                """Sigmoid activation function"""
                # Clip z to prevent overflow
                z = np.clip(z, -500, 500)
                return 1 / (1 + np.exp(-z))
            
            def sigmoid_derivative(self, z):
                """Derivative of sigmoid function"""
                return z * (1 - z)
            
            def forward_propagation(self, X):
                """Forward pass through the network"""
                # Hidden layer
                self.z1 = np.dot(X, self.W1) + self.b1
                self.a1 = self.sigmoid(self.z1)
                
                # Output layer
                self.z2 = np.dot(self.a1, self.W2) + self.b2
                self.a2 = self.sigmoid(self.z2)
                
                return self.a2
            
            def backward_propagation(self, X, y, output):
                """Backward pass - calculate gradients"""
                m = X.shape[0]
                
                # Calculate gradients for output layer
                dz2 = output - y
                dW2 = (1/m) * np.dot(self.a1.T, dz2)
                db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
                
                # Calculate gradients for hidden layer
                dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
                dW1 = (1/m) * np.dot(X.T, dz1)
                db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
                
                return dW1, db1, dW2, db2
            
            def update_parameters(self, dW1, db1, dW2, db2):
                """Update weights and biases"""
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
            
            def compute_cost(self, y_true, y_pred):
                """Compute binary cross-entropy cost"""
                m = y_true.shape[0]
                # Prevent log(0)
                y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
                cost = -(1/m) * np.sum(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
                return cost
            
            def train(self, X, y, epochs=1000):
                """Train the neural network"""
                for epoch in range(epochs):
                    # Forward propagation
                    output = self.forward_propagation(X)
                    
                    # Compute cost
                    cost = self.compute_cost(y, output)
                    self.costs.append(cost)
                    
                    # Backward propagation
                    dW1, db1, dW2, db2 = self.backward_propagation(X, y, output)
                    
                    # Update parameters
                    self.update_parameters(dW1, db1, dW2, db2)
                    
                    if epoch % 100 == 0:
                        print(f"Cost after epoch {epoch}: {cost:.4f}")
            
            def predict(self, X):
                """Make predictions"""
                output = self.forward_propagation(X)
                return (output > 0.5).astype(int)
        
        # Create and train the network
        X_class, y_class, _, _ = self.create_sample_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test = train_test_split(
            X_class, y_class.reshape(-1, 1), test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train network
        print("Training neural network from scratch...")
        nn = SimpleNeuralNetwork(input_size=20, hidden_size=10, output_size=1)
        nn.train(X_train_scaled, y_train, epochs=500)
        
        # Make predictions
        predictions = nn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        print(f"\\nAccuracy: {accuracy:.3f}")
        
        # Plot training cost
        plt.figure(figsize=(10, 6))
        plt.plot(nn.costs)
        plt.title('Training Cost Over Time')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
        
        return nn
    
    def build_tensorflow_models(self):
        """Build various neural network architectures using TensorFlow/Keras"""
        
        print("\\n=== TENSORFLOW/KERAS NEURAL NETWORKS ===")
        print()
        
        X_class, y_class, X_reg, y_reg = self.create_sample_data()
        
        # Prepare classification data
        X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42
        )
        
        # Prepare regression data
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler_class = StandardScaler()
        X_class_train_scaled = scaler_class.fit_transform(X_class_train)
        X_class_test_scaled = scaler_class.transform(X_class_test)
        
        scaler_reg = StandardScaler()
        X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
        X_reg_test_scaled = scaler_reg.transform(X_reg_test)
        
        # 1. SIMPLE FEEDFORWARD NETWORK FOR CLASSIFICATION
        print("1. SIMPLE FEEDFORWARD NETWORK (CLASSIFICATION)")
        print("="*50)
        
        model_simple = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(20,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model_simple.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model_simple.summary()
        
        # Train the model
        history_simple = model_simple.fit(
            X_class_train_scaled, y_class_train,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_acc = model_simple.evaluate(X_class_test_scaled, y_class_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.3f}")
        
        # 2. DEEP NEURAL NETWORK WITH REGULARIZATION
        print("\\n2. DEEP NEURAL NETWORK WITH REGULARIZATION")
        print("="*50)
        
        model_deep = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(20,)),
            layers.Dropout(0.3),  # Dropout for regularization
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),  # Batch normalization
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model_deep.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("Deep model architecture:")
        model_deep.summary()
        
        # Add early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train with callbacks
        history_deep = model_deep.fit(
            X_class_train_scaled, y_class_train,
            batch_size=32,
            epochs=100,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        test_loss_deep, test_acc_deep = model_deep.evaluate(X_class_test_scaled, y_class_test, verbose=0)
        print(f"Deep Network Test Accuracy: {test_acc_deep:.3f}")
        
        # 3. REGRESSION NETWORK
        print("\\n3. NEURAL NETWORK FOR REGRESSION")
        print("="*50)
        
        model_reg = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(15,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # No activation for regression
        ])
        
        model_reg.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Train regression model
        history_reg = model_reg.fit(
            X_reg_train_scaled, y_reg_train,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            verbose=0
        )
        
        # Evaluate
        test_loss_reg, test_mae = model_reg.evaluate(X_reg_test_scaled, y_reg_test, verbose=0)
        
        # Calculate R²
        y_pred_reg = model_reg.predict(X_reg_test_scaled, verbose=0)
        r2_reg = 1 - (np.sum((y_reg_test - y_pred_reg.flatten())**2) / 
                     np.sum((y_reg_test - np.mean(y_reg_test))**2))
        
        print(f"Regression Test MAE: {test_mae:.3f}")
        print(f"Regression Test R²: {r2_reg:.3f}")
        
        # 4. CUSTOM TRAINING LOOP (ADVANCED)
        print("\\n4. CUSTOM TRAINING LOOP")
        print("="*50)
        
        # Create a simple model for custom training
        model_custom = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(20,)),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Define loss and optimizer
        loss_fn = keras.losses.BinaryCrossentropy()
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        
        # Metrics
        train_acc_metric = keras.metrics.BinaryAccuracy()
        val_acc_metric = keras.metrics.BinaryAccuracy()
        
        # Custom training function
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = model_custom(x, training=True)
                loss = loss_fn(y, predictions)
            
            gradients = tape.gradient(loss, model_custom.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_custom.trainable_variables))
            
            train_acc_metric.update_state(y, predictions)
            return loss
        
        @tf.function
        def val_step(x, y):
            predictions = model_custom(x, training=False)
            val_loss = loss_fn(y, predictions)
            val_acc_metric.update_state(y, predictions)
            return val_loss
        
        # Training loop
        print("Training with custom loop...")
        epochs = 20
        train_dataset = tf.data.Dataset.from_tensor_slices((X_class_train_scaled, y_class_train))
        train_dataset = train_dataset.batch(32)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_class_test_scaled, y_class_test))
        val_dataset = val_dataset.batch(32)
        
        for epoch in range(epochs):
            # Training
            train_acc_metric.reset_states()
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                loss = train_step(x_batch, y_batch)
            
            # Validation
            val_acc_metric.reset_states()
            for x_batch, y_batch in val_dataset:
                val_step(x_batch, y_batch)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Acc: {train_acc_metric.result():.3f}, "
                      f"Val Acc: {val_acc_metric.result():.3f}")
        
        # Plot training histories
        self.plot_training_histories(history_simple, history_deep, history_reg)
        
        # Store models
        self.models['simple'] = model_simple
        self.models['deep'] = model_deep
        self.models['regression'] = model_reg
        self.models['custom'] = model_custom
        
        return self.models
    
    def plot_training_histories(self, history_simple, history_deep, history_reg):
        """Plot training histories for comparison"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Simple model
        axes[0,0].plot(history_simple.history['accuracy'], label='Train')
        axes[0,0].plot(history_simple.history['val_accuracy'], label='Validation')
        axes[0,0].set_title('Simple Model - Accuracy')
        axes[0,0].set_xlabel('Epochs')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        axes[1,0].plot(history_simple.history['loss'], label='Train')
        axes[1,0].plot(history_simple.history['val_loss'], label='Validation')
        axes[1,0].set_title('Simple Model - Loss')
        axes[1,0].set_xlabel('Epochs')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Deep model
        axes[0,1].plot(history_deep.history['accuracy'], label='Train')
        axes[0,1].plot(history_deep.history['val_accuracy'], label='Validation')
        axes[0,1].set_title('Deep Model - Accuracy')
        axes[0,1].set_xlabel('Epochs')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        axes[1,1].plot(history_deep.history['loss'], label='Train')
        axes[1,1].plot(history_deep.history['val_loss'], label='Validation')
        axes[1,1].set_title('Deep Model - Loss')
        axes[1,1].set_xlabel('Epochs')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Regression model
        axes[0,2].plot(history_reg.history['mae'], label='Train MAE')
        axes[0,2].plot(history_reg.history['val_mae'], label='Validation MAE')
        axes[0,2].set_title('Regression Model - MAE')
        axes[0,2].set_xlabel('Epochs')
        axes[0,2].set_ylabel('Mean Absolute Error')
        axes[0,2].legend()
        axes[0,2].grid(True)
        
        axes[1,2].plot(history_reg.history['loss'], label='Train')
        axes[1,2].plot(history_reg.history['val_loss'], label='Validation')
        axes[1,2].set_title('Regression Model - Loss')
        axes[1,2].set_xlabel('Epochs')
        axes[1,2].set_ylabel('Mean Squared Error')
        axes[1,2].legend()
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def explain_concepts(self):
        """Explain key neural network concepts"""
        
        print("\\n" + "="*60)
        print("NEURAL NETWORK CONCEPTS EXPLAINED")
        print("="*60)
        
        concepts = {
            "🧠 WHAT IS A NEURAL NETWORK?": [
                "A neural network is inspired by how the human brain works",
                "It consists of interconnected nodes (neurons) organized in layers",
                "Information flows forward through the network to make predictions",
                "The network learns by adjusting connection weights through training"
            ],
            
            "🏗️ NETWORK ARCHITECTURE": [
                "Input Layer: Receives the input features",
                "Hidden Layers: Process information (can have multiple layers)",
                "Output Layer: Produces the final prediction",
                "Deeper networks can learn more complex patterns"
            ],
            
            "⚡ ACTIVATION FUNCTIONS": [
                "ReLU (Rectified Linear Unit): f(x) = max(0, x) - Most common",
                "Sigmoid: f(x) = 1/(1 + e^(-x)) - Outputs between 0 and 1",
                "Tanh: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) - Outputs between -1 and 1",
                "Softmax: Used for multi-class classification output"
            ],
            
            "🎯 TRAINING PROCESS": [
                "1. Forward Propagation: Input flows through network to output",
                "2. Loss Calculation: Compare prediction with actual target",
                "3. Backward Propagation: Calculate gradients of the loss",
                "4. Weight Update: Adjust weights to minimize loss",
                "5. Repeat for many epochs until convergence"
            ],
            
            "📊 LOSS FUNCTIONS": [
                "Mean Squared Error (MSE): For regression problems",
                "Binary Crossentropy: For binary classification",
                "Categorical Crossentropy: For multi-class classification",
                "Sparse Categorical Crossentropy: When classes are integers"
            ],
            
            "🔧 OPTIMIZATION": [
                "SGD (Stochastic Gradient Descent): Basic optimizer",
                "Adam: Adaptive learning rate, very popular",
                "RMSprop: Good for recurrent neural networks",
                "AdaGrad: Adapts learning rate based on parameters"
            ],
            
            "🛡️ REGULARIZATION TECHNIQUES": [
                "Dropout: Randomly ignore neurons during training",
                "Batch Normalization: Normalize inputs to each layer",
                "L1/L2 Regularization: Add penalty for large weights",
                "Early Stopping: Stop training when validation loss increases"
            ],
            
            "🎛️ HYPERPARAMETERS": [
                "Learning Rate: How fast the model learns (0.001 is common)",
                "Batch Size: Number of samples processed together (32, 64, 128)",
                "Epochs: Number of times model sees entire dataset",
                "Hidden Units: Number of neurons in hidden layers",
                "Number of Layers: Depth of the network"
            ],
            
            "⚠️ COMMON PROBLEMS": [
                "Overfitting: Model memorizes training data, poor generalization",
                "Underfitting: Model is too simple to capture patterns",
                "Vanishing Gradients: Gradients become too small in deep networks",
                "Exploding Gradients: Gradients become too large"
            ],
            
            "✅ BEST PRACTICES": [
                "Start simple: Begin with a basic architecture",
                "Scale features: Normalize input data",
                "Use validation data: Monitor overfitting",
                "Experiment: Try different architectures and hyperparameters",
                "Visualize training: Plot loss and accuracy curves"
            ]
        }
        
        for concept, points in concepts.items():
            print(f"\\n{concept}")
            print("-" * len(concept))
            for point in points:
                print(f"• {point}")
        
        print("\\n" + "="*60)
        print("WHEN TO USE NEURAL NETWORKS")
        print("="*60)
        
        use_cases = {
            "✅ GOOD FOR:": [
                "Image recognition and computer vision",
                "Natural language processing",
                "Complex non-linear patterns",
                "Large amounts of data available",
                "High-dimensional data",
                "When accuracy is more important than interpretability"
            ],
            
            "❌ NOT IDEAL FOR:": [
                "Small datasets (less than 1000 samples)",
                "When interpretability is crucial",
                "Simple linear relationships",
                "Limited computational resources",
                "When traditional ML methods work well"
            ]
        }
        
        for category, items in use_cases.items():
            print(f"\\n{category}")
            print("-" * len(category))
            for item in items:
                print(f"• {item}")

# Run the complete neural networks guide
guide = NeuralNetworkGuide()

# Build neural network from scratch
guide.build_basic_neural_network()

# Build TensorFlow models
guide.build_tensorflow_models()

# Explain concepts
guide.explain_concepts()

print("\\n🎉 Neural Networks and Deep Learning Guide Complete!")
print("You now have a solid foundation in neural networks and deep learning.")
print("Next steps: Explore CNNs for images, RNNs for sequences, and transfer learning!")`
  }

  const renderContent = () => {
    switch(activeSection) {
      case 'introduction':
        return (
          <div className="section-content">
            <h2>🤖 What is Machine Learning?</h2>
            
            <div className="intro-section">
              <h3>Understanding Machine Learning</h3>
              <p>
                Machine Learning is a subset of artificial intelligence (AI) that enables computers to learn and 
                make decisions from data without being explicitly programmed. Instead of following pre-written 
                instructions, ML algorithms find patterns in data and use these patterns to make predictions 
                about new, unseen data.
              </p>
            </div>

            <div className="analogy-box">
              <h4>🧠 Human Learning vs Machine Learning</h4>
              <div className="direction-comparison">
                <div className="human-directions">
                  <h5>How Humans Learn</h5>
                  <ul>
                    <li>👁️ Observe examples and patterns</li>
                    <li>🧠 Process and understand relationships</li>
                    <li>💭 Form mental models and rules</li>
                    <li>🎯 Apply knowledge to new situations</li>
                    <li>📚 Learn from mistakes and feedback</li>
                  </ul>
                </div>
                <div className="computer-directions">
                  <h5>How Machines Learn</h5>
                  <ul>
                    <li>📊 Analyze large amounts of training data</li>
                    <li>🔍 Identify statistical patterns and relationships</li>
                    <li>⚙️ Build mathematical models</li>
                    <li>🎯 Make predictions on new data</li>
                    <li>🔄 Improve through more data and feedback</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="concept-deep-dive">
              <h3>🎯 Types of Machine Learning</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>👨‍🏫 Supervised Learning</h4>
                  <p>Learn from labeled examples (input-output pairs)</p>
                  <div className="real-example">
                    <strong>Examples:</strong> Email spam detection, house price prediction, medical diagnosis
                  </div>
                </div>
                <div className="data-type-card">
                  <h4>🔍 Unsupervised Learning</h4>
                  <p>Find hidden patterns in data without labels</p>
                  <div className="real-example">
                    <strong>Examples:</strong> Customer segmentation, data compression, anomaly detection
                  </div>
                </div>
                <div className="data-type-card">
                  <h4>🎮 Reinforcement Learning</h4>
                  <p>Learn through trial and error with rewards/penalties</p>
                  <div className="real-example">
                    <strong>Examples:</strong> Game playing (chess, Go), robotics, autonomous vehicles
                  </div>
                </div>
                <div className="data-type-card">
                  <h4>🧠 Deep Learning</h4>
                  <p>Neural networks with many layers for complex patterns</p>
                  <div className="real-example">
                    <strong>Examples:</strong> Image recognition, natural language processing, speech recognition
                  </div>
                </div>
              </div>
            </div>

            <div className="math-concept">
              <h3>🔄 The Machine Learning Process</h3>
              <div className="translation-process">
                <div className="step">
                  <div className="step-number">1</div>
                  <div>
                    <strong>Data Collection</strong>
                    <p>Gather relevant, high-quality data for your problem</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">2</div>
                  <div>
                    <strong>Data Preprocessing</strong>
                    <p>Clean, transform, and prepare data for modeling</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">3</div>
                  <div>
                    <strong>Model Selection</strong>
                    <p>Choose appropriate algorithm based on problem type</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">4</div>
                  <div>
                    <strong>Training</strong>
                    <p>Feed data to algorithm to learn patterns</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">5</div>
                  <div>
                    <strong>Evaluation</strong>
                    <p>Test model performance on unseen data</p>
                  </div>
                </div>
                <div className="arrow">↓</div>
                <div className="step">
                  <div className="step-number">6</div>
                  <div>
                    <strong>Deployment</strong>
                    <p>Use model to make predictions on new data</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="code-example">
              <h4>Complete ML Example: House Price Prediction</h4>
              <pre>{codeExamples.basicML}</pre>
              <button onClick={() => setExpandedCode(expandedCode === 'basicML' ? null : 'basicML')}>
                {expandedCode === 'basicML' ? 'Hide Code Explanation' : 'Show Detailed Code Explanation'}
              </button>
              {expandedCode === 'basicML' && (
                <div className="code-explanation">
                  <h5>🔍 Code Breakdown:</h5>
                  <ul>
                    <li><strong>Data Creation:</strong> Generate realistic house features and prices</li>
                    <li><strong>Data Exploration:</strong> Understand the dataset structure and statistics</li>
                    <li><strong>Train-Test Split:</strong> Separate data for training and evaluation</li>
                    <li><strong>Feature Scaling:</strong> Normalize features for better model performance</li>
                    <li><strong>Model Training:</strong> Use Linear Regression to learn patterns</li>
                    <li><strong>Evaluation:</strong> Measure model performance with metrics</li>
                    <li><strong>Prediction:</strong> Use trained model on new data</li>
                    <li><strong>Visualization:</strong> Compare actual vs predicted values</li>
                  </ul>
                </div>
              )}
            </div>
          </div>
        )

      case 'mathematics':
        return (
          <div className="section-content">
            <h2>📊 ML Mathematics Foundation</h2>
            
            <div className="intro-section">
              <h3>Why Mathematics Matters in ML</h3>
              <p>
                Mathematics is the foundation of machine learning. While you can use ML libraries without deep math knowledge,
                understanding the underlying mathematics helps you choose the right algorithms, debug problems,
                and optimize performance. Don't worry - we'll explain everything step by step!
              </p>
            </div>

            <div className="concept-deep-dive">
              <h3>🔢 Essential Math Topics</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>📊 Statistics & Probability</h4>
                  <p>Understanding data distributions, correlation, and uncertainty</p>
                  <div className="real-example">Mean, median, standard deviation, confidence intervals, Bayes' theorem</div>
                </div>
                <div className="data-type-card">
                  <h4>🔢 Linear Algebra</h4>
                  <p>Vectors, matrices, and operations on multi-dimensional data</p>
                  <div className="real-example">Matrix multiplication, eigenvalues, principal component analysis</div>
                </div>
                <div className="data-type-card">
                  <h4>📈 Calculus</h4>
                  <p>Optimization and understanding how algorithms learn</p>
                  <div className="real-example">Derivatives, gradients, gradient descent optimization</div>
                </div>
                <div className="data-type-card">
                  <h4>🔍 Discrete Mathematics</h4>
                  <p>Logic, sets, and combinatorics for algorithmic thinking</p>
                  <div className="real-example">Graph theory, information theory, computational complexity</div>
                </div>
              </div>
            </div>

            <div className="math-explanation">
              <h4>📊 Statistics Concepts You Need</h4>
              <div className="code-example">
                <pre>{`# Essential Statistics for ML
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, 1000)  # Mean=100, Std=15

# Descriptive Statistics
print("Descriptive Statistics:")
print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Mode: {stats.mode(data.round())}")
print(f"Standard Deviation: {np.std(data):.2f}")
print(f"Variance: {np.var(data):.2f}")

# Percentiles
print(f"\\n25th Percentile: {np.percentile(data, 25):.2f}")
print(f"75th Percentile: {np.percentile(data, 75):.2f}")
print(f"IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")

# Correlation (between two variables)
x = np.random.normal(50, 10, 1000)
y = 2 * x + np.random.normal(0, 5, 1000)  # y is correlated with x
correlation = np.corrcoef(x, y)[0, 1]
print(f"\\nCorrelation coefficient: {correlation:.3f}")

# Probability distributions
print("\\nProbability Distribution:")
print(f"P(data < 85) = {np.mean(data < 85):.3f}")
print(f"P(85 ≤ data ≤ 115) = {np.mean((data >= 85) & (data <= 115)):.3f}")`}</pre>
              </div>
            </div>

            <div className="math-explanation">
              <h4>🔢 Linear Algebra Essentials</h4>
              <div className="code-example">
                <pre>{`# Linear Algebra for ML
import numpy as np

# Vectors
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

print("Vector Operations:")
print(f"Vector A: {vector_a}")
print(f"Vector B: {vector_b}")
print(f"Dot Product: {np.dot(vector_a, vector_b)}")
print(f"Cross Product: {np.cross(vector_a, vector_b)}")
print(f"Magnitude of A: {np.linalg.norm(vector_a):.3f}")

# Matrices
matrix_A = np.array([[1, 2], [3, 4]])
matrix_B = np.array([[5, 6], [7, 8]])

print("\\nMatrix Operations:")
print(f"Matrix A:\\n{matrix_A}")
print(f"Matrix B:\\n{matrix_B}")
print(f"Matrix Multiplication A @ B:\\n{matrix_A @ matrix_B}")
print(f"Matrix A Transpose:\\n{matrix_A.T}")
print(f"Matrix A Inverse:\\n{np.linalg.inv(matrix_A)}")
print(f"Determinant of A: {np.linalg.det(matrix_A):.3f}")

# Eigenvalues and Eigenvectors (important for PCA)
eigenvalues, eigenvectors = np.linalg.eig(matrix_A)
print(f"\\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:\\n{eigenvectors}")`}</pre>
              </div>
            </div>

            <div className="math-explanation">
              <h4>📈 Calculus for Optimization</h4>
              <p>Most ML algorithms work by minimizing a cost function using calculus-based optimization.</p>
              <div className="code-example">
                <pre>{`# Calculus Concepts in ML
import numpy as np
import matplotlib.pyplot as plt

# Simple function: f(x) = x^2 + 2x + 1
def f(x):
    return x**2 + 2*x + 1

# Derivative: f'(x) = 2x + 2
def f_derivative(x):
    return 2*x + 2

# Gradient Descent Implementation
def gradient_descent(learning_rate=0.1, iterations=50):
    x = 5.0  # Starting point
    history = [x]
    
    for i in range(iterations):
        gradient = f_derivative(x)
        x = x - learning_rate * gradient  # Update rule
        history.append(x)
        
        if i % 10 == 0:
            print(f"Iteration {i}: x = {x:.4f}, f(x) = {f(x):.4f}")
    
    return history

print("Gradient Descent Optimization:")
history = gradient_descent()

print(f"\\nOptimal x: {history[-1]:.4f}")
print(f"Minimum value: {f(history[-1]):.4f}")
print("(Analytical minimum is at x = -1, f(-1) = 0)")

# Visualize the optimization process
x_range = np.linspace(-3, 2, 100)
y_range = [f(x) for x in x_range]

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, 'b-', label='f(x) = x² + 2x + 1')
plt.plot(history, [f(x) for x in history], 'ro-', label='Gradient Descent Path')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.grid(True)
plt.show()`}</pre>
              </div>
            </div>
          </div>
        )

      case 'data-preprocessing':
        return (
          <div className="section-content">
            <h2>🧹 Data Preprocessing</h2>
            
            <div className="intro-section">
              <h3>Why Data Preprocessing Matters</h3>
              <p>
                "Garbage in, garbage out" - this famous saying perfectly captures why data preprocessing is crucial. 
                Real-world data is messy, incomplete, and inconsistent. Before we can apply machine learning algorithms, 
                we need to clean and prepare our data. This step often takes 70-80% of a data scientist's time!
              </p>
            </div>

            <div className="code-example">
              <h4>Complete Data Preprocessing Pipeline</h4>
              <pre>{codeExamples.dataPreprocessing}</pre>
            </div>

            <div className="concept-deep-dive">
              <h3>🔧 Preprocessing Techniques</h3>
              <div className="data-types-grid">
                <div className="data-type-card">
                  <h4>🧽 Data Cleaning</h4>
                  <p>Handle missing values, outliers, and inconsistencies</p>
                  <div className="real-example">Remove duplicates, fill missing values, cap outliers</div>
                </div>
                <div className="data-type-card">
                  <h4>🔄 Data Transformation</h4>
                  <p>Convert data into suitable format for ML algorithms</p>
                  <div className="real-example">Scaling, normalization, encoding categorical variables</div>
                </div>
                <div className="data-type-card">
                  <h4>⚙️ Feature Engineering</h4>
                  <p>Create new features from existing ones</p>
                  <div className="real-example">Polynomial features, interaction terms, binning</div>
                </div>
                <div className="data-type-card">
                  <h4>🎯 Feature Selection</h4>
                  <p>Choose the most relevant features for your model</p>
                  <div className="real-example">Correlation analysis, recursive feature elimination</div>
                </div>
              </div>
            </div>
          </div>
        )

      case 'supervised-learning':
        return (
          <div className="section-content">
            <h2>👨‍🏫 Supervised Learning</h2>
            
            <div className="intro-section">
              <h3>Understanding Supervised Learning</h3>
              <p>
                Supervised learning is like learning with a teacher. You have input data (features) and the correct 
                answers (labels/targets). The algorithm learns the relationship between inputs and outputs, 
                then uses this knowledge to make predictions on new, unseen data.
              </p>
            </div>

            <div className="code-example">
              <h4>Complete Supervised Learning Guide</h4>
              <pre>{codeExamples.supervisedLearning}</pre>
            </div>

            <div className="concept-deep-dive">
              <h3>🎯 Types of Supervised Learning</h3>
              <div className="direction-comparison">
                <div className="human-directions">
                  <h5>🎯 Classification</h5>
                  <ul>
                    <li>Predict categories or classes</li>
                    <li>Output is discrete (0/1, cat/dog/bird)</li>
                    <li>Examples: Email spam detection, image recognition</li>
                    <li>Metrics: Accuracy, precision, recall, F1-score</li>
                  </ul>
                </div>
                <div className="computer-directions">
                  <h5>📈 Regression</h5>
                  <ul>
                    <li>Predict continuous numerical values</li>
                    <li>Output is a number (price, temperature)</li>
                    <li>Examples: House price prediction, stock prices</li>
                    <li>Metrics: MSE, RMSE, MAE, R²</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )

      case 'neural-networks':
        return (
          <div className="section-content">
            <h2>🧠 Neural Networks & Deep Learning</h2>
            
            <div className="intro-section">
              <h3>Understanding Neural Networks</h3>
              <p>
                Neural networks are inspired by how the human brain processes information. They consist of 
                interconnected nodes (neurons) that work together to learn complex patterns from data. 
                When we have multiple hidden layers, we call it "deep learning."
              </p>
            </div>

            <div className="code-example">
              <h4>Complete Neural Networks Guide</h4>
              <pre>{codeExamples.neuralNetworks}</pre>
            </div>

            <div className="concept-deep-dive">
              <h3>🏗️ Neural Network Architecture</h3>
              <div className="data-level">
                <h4>Input Layer → Hidden Layer(s) → Output Layer</h4>
                <p>Each layer transforms the data, learning increasingly complex features:</p>
                <ul>
                  <li><strong>Input Layer:</strong> Receives raw data (pixels, features, text)</li>
                  <li><strong>Hidden Layers:</strong> Extract features and patterns (edges, shapes, concepts)</li>
                  <li><strong>Output Layer:</strong> Makes final predictions (classes, values)</li>
                </ul>
              </div>
            </div>
          </div>
        )

      default:
        return <div>Select a topic from the sidebar to begin your machine learning journey!</div>
    }
  }

  return (
    <div className="page">
      <div className="learning-container">
        <div className="sidebar">
          <h3>🤖 Complete ML Guide</h3>
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

export default MachineLearningComplete