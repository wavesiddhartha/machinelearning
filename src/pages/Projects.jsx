function Projects() {
  return (
    <div className="page">
      <div className="content">
        <h1>🚀 AI Model Building Projects</h1>
        
        <section className="projects-intro">
          <h2>Build Real AI Applications</h2>
          <p>Learning by doing is the best way to master AI and machine learning. These projects will take you from beginner to expert, with complete code examples and mathematical explanations.</p>
        </section>

        <section className="project-categories">
          <h2>📚 Project Categories</h2>
          
          <div className="category-grid">
            <div className="category-card beginner">
              <h3>🌱 Beginner Projects</h3>
              <p>Perfect for getting started with ML fundamentals</p>
              <ul>
                <li>House Price Predictor</li>
                <li>Iris Flower Classifier</li>
                <li>Customer Segmentation</li>
                <li>Movie Recommender</li>
              </ul>
            </div>

            <div className="category-card intermediate">
              <h3>🚀 Intermediate Projects</h3>
              <p>Dive deeper into real-world applications</p>
              <ul>
                <li>Stock Price Predictor</li>
                <li>Image Classifier</li>
                <li>Sentiment Analysis</li>
                <li>Fraud Detection System</li>
              </ul>
            </div>

            <div className="category-card advanced">
              <h3>🔥 Advanced Projects</h3>
              <p>Cutting-edge AI applications</p>
              <ul>
                <li>Chatbot with Transformers</li>
                <li>Object Detection</li>
                <li>Time Series Forecasting</li>
                <li>Generative Art with GANs</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="featured-project">
          <h2>⭐ Featured Project: Complete ML Pipeline</h2>
          
          <div className="project-overview">
            <h3>🎯 Project: Customer Churn Prediction</h3>
            <p>Build an end-to-end machine learning system to predict which customers are likely to leave a subscription service.</p>
            
            <div className="project-details">
              <div className="project-info">
                <h4>📋 What You'll Learn:</h4>
                <ul>
                  <li>Data exploration and visualization</li>
                  <li>Feature engineering techniques</li>
                  <li>Model selection and evaluation</li>
                  <li>Hyperparameter tuning</li>
                  <li>Model deployment basics</li>
                </ul>
              </div>
              
              <div className="project-tech">
                <h4>🛠️ Technologies Used:</h4>
                <ul>
                  <li>Python & Jupyter Notebooks</li>
                  <li>Pandas & NumPy</li>
                  <li>Scikit-learn</li>
                  <li>Matplotlib & Seaborn</li>
                  <li>Flask (for deployment)</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="code-example">
            <h4>Complete Implementation</h4>
            <pre><code>{`import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionPipeline:
    """Complete ML pipeline for customer churn prediction"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        
    def create_sample_data(self, n_samples=5000):
        """Create realistic sample churn dataset"""
        np.random.seed(42)
        
        # Generate customer features
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(40, 15, n_samples).clip(18, 80),
            'tenure_months': np.random.exponential(24, n_samples).clip(1, 100),
            'monthly_charges': np.random.normal(65, 20, n_samples).clip(15, 150),
            'total_charges': np.random.normal(1500, 800, n_samples).clip(50, 8000),
            'contract_type': np.random.choice(['Month-to-month', '1 year', '2 year'], n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        }
        
        # Create realistic churn based on features
        churn_probability = (
            0.3 * (data['contract_type'] == 'Month-to-month') +
            0.2 * (data['monthly_charges'] > 70) +
            0.15 * (data['tenure_months'] < 12) +
            0.1 * (data['payment_method'] == 'Electronic check') +
            0.1 * (data['online_security'] == 'No') +
            0.05 * data['senior_citizen'] +
            np.random.normal(0, 0.1, n_samples)
        ).clip(0, 1)
        
        data['churn'] = np.random.binomial(1, churn_probability, n_samples)
        
        return pd.DataFrame(data)
    
    def load_and_explore_data(self, df=None):
        """Load and perform initial data exploration"""
        if df is None:
            print("Creating sample dataset...")
            df = self.create_sample_data()
        
        print("=" * 60)
        print("DATA EXPLORATION")
        print("=" * 60)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\\nColumn types:")
        print(df.dtypes)
        
        print(f"\\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")
        
        print(f"\\nChurn distribution:")
        churn_counts = df['churn'].value_counts()
        churn_percentages = df['churn'].value_counts(normalize=True) * 100
        for label, count in churn_counts.items():
            print(f"  {label}: {count} ({churn_percentages[label]:.1f}%)")
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Churn distribution
        df['churn'].value_counts().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Churn Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Age distribution by churn
        df.boxplot(column='age', by='churn', ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Churn')
        
        # 3. Monthly charges by churn
        df.boxplot(column='monthly_charges', by='churn', ax=axes[0, 2])
        axes[0, 2].set_title('Monthly Charges by Churn')
        
        # 4. Tenure distribution
        axes[1, 0].hist(df['tenure_months'], bins=30, alpha=0.7)
        axes[1, 0].set_title('Tenure Distribution')
        axes[1, 0].set_xlabel('Tenure (months)')
        
        # 5. Contract type vs churn
        contract_churn = pd.crosstab(df['contract_type'], df['churn'], normalize='index')
        contract_churn.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Churn Rate by Contract Type')
        axes[1, 1].legend(['No Churn', 'Churn'])
        
        # 6. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 2])
        axes[1, 2].set_title('Correlation Heatmap')
        
        plt.tight_layout()
        # plt.show()
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the dataset"""
        print("\\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)
        
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols.drop('customer_id')  # Don't encode ID
        
        print(f"Encoding {len(categorical_cols)} categorical variables...")
        for col in categorical_cols:
            if col != 'churn':  # Don't encode target variable
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        # Feature engineering
        print("\\nCreating new features...")
        df_processed['charges_per_month_tenure'] = df_processed['total_charges'] / (df_processed['tenure_months'] + 1)
        df_processed['is_new_customer'] = (df_processed['tenure_months'] < 12).astype(int)
        df_processed['high_value_customer'] = (df_processed['monthly_charges'] > df_processed['monthly_charges'].quantile(0.75)).astype(int)
        
        # Create age groups
        df_processed['age_group'] = pd.cut(df_processed['age'], 
                                         bins=[0, 30, 50, 70, 100], 
                                         labels=['Young', 'Middle', 'Senior', 'Elderly'])
        df_processed['age_group'] = LabelEncoder().fit_transform(df_processed['age_group'].astype(str))
        
        print(f"Final dataset shape: {df_processed.shape}")
        print(f"New features created: charges_per_month_tenure, is_new_customer, high_value_customer, age_group")
        
        return df_processed
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        # Exclude non-predictive columns
        exclude_cols = ['customer_id', 'churn']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['churn']
        
        print(f"\\nFeatures selected: {len(feature_cols)}")
        print(f"Feature names: {feature_cols}")
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train multiple models and compare performance"""
        print("\\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Define models to try
        model_configs = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        best_score = 0
        results = {}
        
        for name, config in model_configs.items():
            print(f"\\nTraining {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store results
            results[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_,
                'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
            }
            
            print(f"  Best CV Score: {grid_search.best_score_:.4f} (+/- {results[name]['cv_std']*2:.4f})")
            print(f"  Best Params: {grid_search.best_params_}")
            
            # Update best model
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name
        
        self.models = results
        print(f"\\nBest model: {self.best_model_name} with CV score: {best_score:.4f}")
        
        return results
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model on test set"""
        print("\\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\\nConfusion Matrix:")
        print(cm)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"\\nROC AUC Score: {roc_auc:.4f}")
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = X_test.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\\nTop 10 Most Important Features:")
            print(importance_df.head(10))
            self.feature_importance = importance_df
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        
        # 3. Prediction Probability Distribution
        axes[1, 0].hist(y_pred_proba[y_test == 0], alpha=0.7, bins=30, 
                       label='No Churn', density=True)
        axes[1, 0].hist(y_pred_proba[y_test == 1], alpha=0.7, bins=30, 
                       label='Churn', density=True)
        axes[1, 0].set_xlabel('Prediction Probability')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].legend()
        
        # 4. Feature Importance
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 10 Feature Importance')
        
        plt.tight_layout()
        # plt.show()
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("🚀 STARTING CUSTOMER CHURN PREDICTION PIPELINE")
        print("=" * 80)
        
        # 1. Load and explore data
        df = self.load_and_explore_data()
        
        # 2. Preprocess data
        df_processed = self.preprocess_data(df)
        
        # 3. Prepare features
        X, y = self.prepare_features(df_processed)
        
        # 4. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 5. Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to keep column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        print(f"\\nTrain set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        
        # 6. Train models
        self.train_models(X_train_scaled, y_train)
        
        # 7. Evaluate best model
        results = self.evaluate_model(X_test_scaled, y_test)
        
        print("\\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return {
            'model': self.best_model,
            'scaler': self.scaler,
            'results': results,
            'feature_importance': self.feature_importance
        }

# Run the complete pipeline
if __name__ == "__main__":
    pipeline = ChurnPredictionPipeline()
    final_results = pipeline.run_complete_pipeline()
    
    print("\\n🎯 KEY INSIGHTS:")
    print("1. The model can predict customer churn with high accuracy")
    print("2. Most important factors: contract type, tenure, monthly charges")
    print("3. Month-to-month customers are at highest risk")
    print("4. Early intervention strategies should target new customers")
    print("\\n📊 Business Impact:")
    print("- Identify at-risk customers before they churn")
    print("- Target retention campaigns more effectively")
    print("- Estimate revenue impact of churn")
    print("- Optimize pricing and contract strategies")`}</code></pre>
          </div>
        </section>

        <section className="project-ideas">
          <h2>💡 More Project Ideas</h2>
          
          <div className="ideas-grid">
            <div className="idea-card">
              <h3>🏠 Real Estate Price Predictor</h3>
              <p><strong>Difficulty:</strong> Beginner-Intermediate</p>
              <p><strong>Skills:</strong> Regression, feature engineering, web scraping</p>
              <p>Build a model to predict house prices based on location, size, amenities, and market trends.</p>
            </div>

            <div className="idea-card">
              <h3>🎬 Movie Recommendation System</h3>
              <p><strong>Difficulty:</strong> Intermediate</p>
              <p><strong>Skills:</strong> Collaborative filtering, matrix factorization, deep learning</p>
              <p>Create a Netflix-style recommendation engine using collaborative and content-based filtering.</p>
            </div>

            <div className="idea-card">
              <h3>🛡️ Fraud Detection System</h3>
              <p><strong>Difficulty:</strong> Intermediate-Advanced</p>
              <p><strong>Skills:</strong> Anomaly detection, imbalanced learning, real-time processing</p>
              <p>Detect fraudulent transactions using machine learning and streaming data.</p>
            </div>

            <div className="idea-card">
              <h3>🤖 AI Chatbot</h3>
              <p><strong>Difficulty:</strong> Advanced</p>
              <p><strong>Skills:</strong> NLP, transformers, conversational AI, deployment</p>
              <p>Build an intelligent chatbot using modern NLP techniques and transformer models.</p>
            </div>

            <div className="idea-card">
              <h3>👁️ Computer Vision App</h3>
              <p><strong>Difficulty:</strong> Advanced</p>
              <p><strong>Skills:</strong> CNNs, object detection, image processing</p>
              <p>Create an app that can identify objects, faces, or analyze medical images.</p>
            </div>

            <div className="idea-card">
              <h3>📈 Algorithmic Trading Bot</h3>
              <p><strong>Difficulty:</strong> Expert</p>
              <p><strong>Skills:</strong> Time series, reinforcement learning, quantitative finance</p>
              <p>Develop an AI-powered trading system that makes investment decisions.</p>
            </div>
          </div>
        </section>

        <section className="project-resources">
          <h2>📚 Project Resources</h2>
          
          <div className="resources-grid">
            <div className="resource-category">
              <h3>🗄️ Datasets</h3>
              <ul>
                <li>Kaggle (kaggle.com/datasets)</li>
                <li>UCI ML Repository</li>
                <li>Google Dataset Search</li>
                <li>Government Open Data</li>
                <li>Company APIs (Twitter, Reddit, etc.)</li>
              </ul>
            </div>

            <div className="resource-category">
              <h3>☁️ Cloud Platforms</h3>
              <ul>
                <li>Google Colab (Free GPU/TPU)</li>
                <li>AWS SageMaker</li>
                <li>Google Cloud AI Platform</li>
                <li>Microsoft Azure ML</li>
                <li>Paperspace Gradient</li>
              </ul>
            </div>

            <div className="resource-category">
              <h3>📊 Visualization Tools</h3>
              <ul>
                <li>Matplotlib & Seaborn</li>
                <li>Plotly & Dash</li>
                <li>Streamlit</li>
                <li>Tableau Public</li>
                <li>Power BI</li>
              </ul>
            </div>

            <div className="resource-category">
              <h3>🚀 Deployment Platforms</h3>
              <ul>
                <li>Streamlit Sharing</li>
                <li>Heroku</li>
                <li>AWS EC2/Lambda</li>
                <li>Google Cloud Run</li>
                <li>Docker & Kubernetes</li>
              </ul>
            </div>
          </div>
        </section>

        <div className="project-tips">
          <h2>🎯 Tips for Successful Projects</h2>
          
          <div className="tips-list">
            <div className="tip">
              <h4>1. Start Simple, Then Iterate</h4>
              <p>Begin with a basic model, then gradually add complexity and improvements.</p>
            </div>
            
            <div className="tip">
              <h4>2. Focus on the Problem, Not Just the Algorithm</h4>
              <p>Understand the business context and define clear success metrics.</p>
            </div>
            
            <div className="tip">
              <h4>3. Document Everything</h4>
              <p>Keep detailed notes of your process, decisions, and results.</p>
            </div>
            
            <div className="tip">
              <h4>4. Build an End-to-End Pipeline</h4>
              <p>Include data collection, preprocessing, training, evaluation, and deployment.</p>
            </div>
            
            <div className="tip">
              <h4>5. Share Your Work</h4>
              <p>Create a portfolio on GitHub, write blog posts, and present your findings.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Projects