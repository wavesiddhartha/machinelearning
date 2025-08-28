import { useState } from 'react'

function NaturalLanguageProcessingComplete() {
  const [activeSection, setActiveSection] = useState(0)
  const [expandedCode, setExpandedCode] = useState({})

  const toggleCode = (sectionId, codeId) => {
    const key = `${sectionId}-${codeId}`
    setExpandedCode(prev => ({
      ...prev,
      [key]: !prev[key]
    }))
  }

  const sections = [
    {
      id: 'nlp-fundamentals',
      title: 'NLP Fundamentals & Text Processing',
      icon: '📝',
      description: 'Master the foundations of natural language processing and text analysis',
      content: `
        Natural Language Processing enables computers to understand, interpret, and generate human language.
        Learn essential text processing techniques, tokenization, and linguistic analysis fundamentals.
      `,
      keyTopics: [
        'Text Preprocessing and Cleaning',
        'Tokenization and Stemming',
        'Part-of-Speech Tagging',
        'Named Entity Recognition (NER)',
        'Regular Expressions for Text',
        'Language Detection',
        'Text Normalization',
        'Corpus Processing with NLTK and spaCy'
      ],
      codeExamples: [
        {
          title: 'Text Preprocessing with NLTK and spaCy',
          description: 'Essential text preprocessing techniques for NLP',
          code: `import nltk
import spacy
import re
import string
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Load spaCy model (install with: python -m spacy download en_core_web_sm)
nlp = spacy.load('en_core_web_sm')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.PorterStemmer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        
    def clean_text(self, text):
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (for social media)
        text = re.sub(r'@\\w+|#\\w+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """Tokenize text using NLTK"""
        # Word tokenization
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Remove single characters
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def pos_tagging(self, tokens):
        """Part-of-speech tagging"""
        return nltk.pos_tag(tokens)
    
    def extract_entities_spacy(self, text):
        """Extract named entities using spaCy"""
        doc = nlp(text)
        entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        return entities
    
    def extract_noun_phrases(self, text):
        """Extract noun phrases using spaCy"""
        doc = nlp(text)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return noun_phrases
    
    def dependency_parsing(self, text):
        """Analyze sentence dependencies"""
        doc = nlp(text)
        dependencies = []
        for token in doc:
            dependencies.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'head': token.head.text,
                'children': [child.text for child in token.children]
            })
        return dependencies
    
    def analyze_text_statistics(self, text):
        """Comprehensive text analysis"""
        # Clean and tokenize
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        
        # Basic statistics
        stats = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(nltk.sent_tokenize(text)),
            'unique_words': len(set(tokens)),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        }
        
        # Most common words
        word_freq = Counter(tokens)
        stats['most_common_words'] = word_freq.most_common(10)
        
        # POS tag distribution
        pos_tags = self.pos_tagging(tokens)
        pos_freq = Counter([tag for word, tag in pos_tags])
        stats['pos_distribution'] = pos_freq.most_common()
        
        return stats
    
    def create_wordcloud(self, text, save_path=None):
        """Generate word cloud"""
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        text_for_cloud = ' '.join(tokens)
        
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text_for_cloud)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
        return wordcloud

class SentimentAnalyzer:
    def __init__(self):
        from textblob import TextBlob
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def textblob_sentiment(self, text):
        """Sentiment analysis using TextBlob"""
        from textblob import TextBlob
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity,  # 0 to 1
            'sentiment': 'positive' if blob.sentiment.polarity > 0 else 'negative' if blob.sentiment.polarity < 0 else 'neutral'
        }
    
    def vader_sentiment(self, text):
        """Sentiment analysis using VADER"""
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'sentiment': sentiment
        }
    
    def analyze_sentiment_trends(self, texts):
        """Analyze sentiment trends across multiple texts"""
        results = []
        
        for i, text in enumerate(texts):
            textblob_result = self.textblob_sentiment(text)
            vader_result = self.vader_sentiment(text)
            
            results.append({
                'text_id': i,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'textblob_polarity': textblob_result['polarity'],
                'textblob_sentiment': textblob_result['sentiment'],
                'vader_compound': vader_result['compound'],
                'vader_sentiment': vader_result['sentiment']
            })
        
        return pd.DataFrame(results)

# Comprehensive NLP demonstration
def demonstrate_nlp_fundamentals():
    # Sample texts for analysis
    sample_texts = [
        "Natural Language Processing is an amazing field that combines linguistics and computer science. I love working with text data and discovering insights!",
        "The weather today is terrible. It's raining heavily and I'm stuck inside. This is really frustrating.",
        "Machine learning models for NLP have improved significantly in recent years. Transformers and BERT have revolutionized the field.",
        "Python is an excellent programming language for data science and NLP. Libraries like NLTK, spaCy, and scikit-learn make it easy to work with text.",
        "The movie was absolutely fantastic! Great acting, wonderful plot, and amazing special effects. Highly recommended!"
    ]
    
    # Initialize processors
    preprocessor = TextPreprocessor()
    sentiment_analyzer = SentimentAnalyzer()
    
    print("=== NLP Fundamentals Demonstration ===\\n")
    
    # Analyze first text in detail
    sample_text = sample_texts[0]
    print(f"Analyzing: {sample_text}\\n")
    
    # 1. Text preprocessing
    cleaned = preprocessor.clean_text(sample_text)
    tokens = preprocessor.tokenize_text(cleaned)
    stemmed = preprocessor.stem_tokens(tokens)
    lemmatized = preprocessor.lemmatize_tokens(tokens)
    
    print("1. Text Preprocessing:")
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned}")
    print(f"Tokens: {tokens}")
    print(f"Stemmed: {stemmed}")
    print(f"Lemmatized: {lemmatized}\\n")
    
    # 2. POS tagging
    pos_tags = preprocessor.pos_tagging(tokens)
    print("2. Part-of-Speech Tagging:")
    for word, tag in pos_tags:
        print(f"{word}: {tag}")
    print()
    
    # 3. Named Entity Recognition
    entities = preprocessor.extract_entities_spacy(sample_text)
    print("3. Named Entities:")
    for entity, label, start, end in entities:
        print(f"{entity} ({label}) at position {start}-{end}")
    print()
    
    # 4. Text statistics
    stats = preprocessor.analyze_text_statistics(sample_text)
    print("4. Text Statistics:")
    for key, value in stats.items():
        if key != 'most_common_words' and key != 'pos_distribution':
            print(f"{key}: {value}")
    print(f"Most common words: {stats['most_common_words'][:5]}")
    print()
    
    # 5. Sentiment analysis
    print("5. Sentiment Analysis:")
    for i, text in enumerate(sample_texts):
        textblob_sentiment = sentiment_analyzer.textblob_sentiment(text)
        vader_sentiment = sentiment_analyzer.vader_sentiment(text)
        
        print(f"Text {i+1}: {text[:50]}...")
        print(f"  TextBlob: {textblob_sentiment['sentiment']} (polarity: {textblob_sentiment['polarity']:.3f})")
        print(f"  VADER: {vader_sentiment['sentiment']} (compound: {vader_sentiment['compound']:.3f})")
        print()
    
    # 6. Sentiment trends analysis
    sentiment_df = sentiment_analyzer.analyze_sentiment_trends(sample_texts)
    print("6. Sentiment Trends Summary:")
    print(f"Average TextBlob Polarity: {sentiment_df['textblob_polarity'].mean():.3f}")
    print(f"Average VADER Compound: {sentiment_df['vader_compound'].mean():.3f}")
    
    # Count sentiment distribution
    textblob_counts = sentiment_df['textblob_sentiment'].value_counts()
    vader_counts = sentiment_df['vader_sentiment'].value_counts()
    
    print("\\nSentiment Distribution:")
    print("TextBlob:", dict(textblob_counts))
    print("VADER:", dict(vader_counts))
    
    return {
        'preprocessor': preprocessor,
        'sentiment_analyzer': sentiment_analyzer,
        'sample_analysis': stats,
        'sentiment_trends': sentiment_df
    }

# Run demonstration
# results = demonstrate_nlp_fundamentals()`
        },
        {
          title: 'Advanced Text Processing & Feature Engineering',
          description: 'N-grams, TF-IDF, and advanced text feature extraction',
          code: `from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedTextProcessor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.lda_model = None
        
    def generate_ngrams(self, text, n=2):
        """Generate n-grams from text"""
        from nltk.util import ngrams
        tokens = nltk.word_tokenize(text.lower())
        n_grams = list(ngrams(tokens, n))
        return [' '.join(gram) for gram in n_grams]
    
    def extract_tfidf_features(self, documents, max_features=1000, ngram_range=(1, 2)):
        """Extract TF-IDF features from documents"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\\b[a-zA-Z]{2,}\\b'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        return tfidf_matrix, feature_names
    
    def get_top_tfidf_words(self, documents, top_k=20):
        """Get top TF-IDF words across all documents"""
        tfidf_matrix, feature_names = self.extract_tfidf_features(documents)
        
        # Sum TF-IDF scores across all documents
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        
        # Get top features
        top_indices = mean_scores.argsort()[-top_k:][::-1]
        top_words = [(feature_names[i], mean_scores[i]) for i in top_indices]
        
        return top_words
    
    def document_similarity(self, documents):
        """Calculate document similarity using TF-IDF and cosine similarity"""
        tfidf_matrix, _ = self.extract_tfidf_features(documents)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix
    
    def topic_modeling_lda(self, documents, n_topics=5, max_features=1000):
        """Perform topic modeling using Latent Dirichlet Allocation"""
        # Prepare count vectorizer (LDA works better with raw term counts)
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\\b[a-zA-Z]{2,}\\b'
        )
        
        doc_term_matrix = self.count_vectorizer.fit_transform(documents)
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10,
            learning_method='online',
            learning_offset=50.0
        )
        
        doc_topic_matrix = self.lda_model.fit_transform(doc_term_matrix)
        
        return doc_topic_matrix
    
    def get_topic_words(self, n_words=10):
        """Get top words for each topic"""
        if self.lda_model is None or self.count_vectorizer is None:
            raise ValueError("Run topic_modeling_lda first")
        
        feature_names = self.count_vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            word_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': word_weights
            })
        
        return topics
    
    def extract_keywords_textrank(self, text, num_keywords=10):
        """Extract keywords using TextRank algorithm"""
        # Simple TextRank implementation
        sentences = nltk.sent_tokenize(text)
        
        # Create TF-IDF matrix for sentences
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Simple PageRank-like algorithm
            scores = np.ones(len(sentences))
            
            for _ in range(50):  # iterations
                new_scores = np.ones(len(sentences))
                for i in range(len(sentences)):
                    for j in range(len(sentences)):
                        if i != j and similarity_matrix[i][j] > 0:
                            new_scores[i] += similarity_matrix[i][j] * scores[j]
                scores = new_scores
            
            # Extract keywords from top sentences
            top_sentence_indices = scores.argsort()[-3:][::-1]
            
            keywords = []
            for idx in top_sentence_indices:
                sentence_words = nltk.word_tokenize(sentences[idx].lower())
                sentence_words = [w for w in sentence_words if w.isalpha() and len(w) > 2]
                keywords.extend(sentence_words)
            
            # Count and return top keywords
            keyword_freq = Counter(keywords)
            return keyword_freq.most_common(num_keywords)
            
        except ValueError:
            # Fallback to simple frequency-based extraction
            words = nltk.word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and len(w) > 2]
            word_freq = Counter(words)
            return word_freq.most_common(num_keywords)
    
    def text_classification_features(self, documents, labels=None):
        """Extract comprehensive features for text classification"""
        features = []
        
        for doc in documents:
            doc_features = {}
            
            # Basic features
            doc_features['length'] = len(doc)
            doc_features['word_count'] = len(doc.split())
            doc_features['sentence_count'] = len(nltk.sent_tokenize(doc))
            doc_features['avg_word_length'] = np.mean([len(word) for word in doc.split()])
            
            # Punctuation features
            doc_features['exclamation_count'] = doc.count('!')
            doc_features['question_count'] = doc.count('?')
            doc_features['comma_count'] = doc.count(',')
            doc_features['period_count'] = doc.count('.')
            
            # Capital letters
            doc_features['uppercase_count'] = sum(1 for c in doc if c.isupper())
            doc_features['uppercase_ratio'] = doc_features['uppercase_count'] / len(doc) if len(doc) > 0 else 0
            
            # POS features
            tokens = nltk.word_tokenize(doc)
            pos_tags = nltk.pos_tag(tokens)
            pos_counts = Counter([tag for word, tag in pos_tags])
            
            # Add POS ratios
            total_words = len(tokens)
            doc_features['noun_ratio'] = sum(count for tag, count in pos_counts.items() if tag.startswith('N')) / total_words if total_words > 0 else 0
            doc_features['verb_ratio'] = sum(count for tag, count in pos_counts.items() if tag.startswith('V')) / total_words if total_words > 0 else 0
            doc_features['adj_ratio'] = sum(count for tag, count in pos_counts.items() if tag.startswith('J')) / total_words if total_words > 0 else 0
            
            features.append(doc_features)
        
        return pd.DataFrame(features)
    
    def visualize_topics(self, documents, n_topics=5):
        """Visualize topic modeling results"""
        # Perform topic modeling
        doc_topic_matrix = self.topic_modeling_lda(documents, n_topics)
        topics = self.get_topic_words()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Topic word clouds
        ax1 = axes[0, 0]
        topic_words = [' '.join(topic['words']) for topic in topics]
        ax1.text(0.5, 0.5, '\\n'.join([f"Topic {i+1}: {words[:50]}..." 
                                     for i, words in enumerate(topic_words)]), 
                ha='center', va='center', transform=ax1.transAxes, fontsize=8)
        ax1.set_title('Top Topic Words')
        ax1.axis('off')
        
        # 2. Document-Topic distribution
        ax2 = axes[0, 1]
        doc_topic_df = pd.DataFrame(doc_topic_matrix, columns=[f'Topic {i+1}' for i in range(n_topics)])
        sns.heatmap(doc_topic_df.head(10), annot=True, cmap='viridis', ax=ax2)
        ax2.set_title('Document-Topic Distribution (First 10 docs)')
        
        # 3. Topic prevalence
        ax3 = axes[1, 0]
        topic_prevalence = doc_topic_matrix.mean(axis=0)
        ax3.bar(range(n_topics), topic_prevalence)
        ax3.set_title('Topic Prevalence')
        ax3.set_xlabel('Topic')
        ax3.set_ylabel('Average Probability')
        
        # 4. TF-IDF top words
        ax4 = axes[1, 1]
        top_tfidf = self.get_top_tfidf_words(documents, top_k=10)
        words, scores = zip(*top_tfidf)
        ax4.barh(range(len(words)), scores)
        ax4.set_yticks(range(len(words)))
        ax4.set_yticklabels(words)
        ax4.set_title('Top TF-IDF Words')
        ax4.set_xlabel('TF-IDF Score')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Demonstration function
def demonstrate_advanced_nlp():
    # Sample document collection
    documents = [
        "Machine learning algorithms are transforming the way we process and analyze data. Deep learning neural networks have shown remarkable success in various applications.",
        "Natural language processing enables computers to understand human language. Text mining and sentiment analysis are popular NLP applications in business intelligence.",
        "Python programming language offers excellent libraries for data science. NumPy, pandas, and scikit-learn are essential tools for machine learning practitioners.",
        "Artificial intelligence is revolutionizing healthcare. Medical diagnosis, drug discovery, and personalized treatment are benefiting from AI technologies.",
        "Computer vision applications include image recognition, object detection, and autonomous vehicles. Convolutional neural networks are the backbone of modern computer vision systems.",
        "Big data analytics requires efficient processing of large datasets. Distributed computing frameworks like Spark and Hadoop enable scalable data processing.",
        "Cloud computing provides on-demand access to computing resources. Amazon AWS, Google Cloud, and Microsoft Azure are leading cloud service providers.",
        "Cybersecurity is critical in the digital age. Encryption, intrusion detection, and vulnerability assessment are essential security measures."
    ]
    
    processor = AdvancedTextProcessor()
    
    print("=== Advanced NLP Processing Demonstration ===\\n")
    
    # 1. N-grams analysis
    sample_text = documents[0]
    bigrams = processor.generate_ngrams(sample_text, n=2)
    trigrams = processor.generate_ngrams(sample_text, n=3)
    
    print("1. N-grams Analysis:")
    print(f"Sample text: {sample_text}")
    print(f"Bigrams: {bigrams[:5]}")
    print(f"Trigrams: {trigrams[:3]}")
    print()
    
    # 2. TF-IDF analysis
    top_tfidf = processor.get_top_tfidf_words(documents, top_k=15)
    print("2. Top TF-IDF Words:")
    for word, score in top_tfidf:
        print(f"{word}: {score:.4f}")
    print()
    
    # 3. Document similarity
    similarity_matrix = processor.document_similarity(documents)
    print("3. Document Similarity Matrix (first 3x3):")
    print(similarity_matrix[:3, :3])
    print()
    
    # 4. Topic modeling
    doc_topic_matrix = processor.topic_modeling_lda(documents, n_topics=4)
    topics = processor.get_topic_words(n_words=8)
    
    print("4. Topic Modeling Results:")
    for topic in topics:
        print(f"Topic {topic['topic_id'] + 1}: {', '.join(topic['words'])}")
    print()
    
    # 5. Keyword extraction
    keywords = processor.extract_keywords_textrank(documents[0], num_keywords=8)
    print("5. Extracted Keywords (TextRank):")
    for keyword, freq in keywords:
        print(f"{keyword}: {freq}")
    print()
    
    # 6. Feature engineering
    features_df = processor.text_classification_features(documents[:5])
    print("6. Text Classification Features (first 5 documents):")
    print(features_df.head())
    print()
    
    # 7. Visualize results
    print("7. Generating visualizations...")
    # processor.visualize_topics(documents, n_topics=4)
    
    return {
        'processor': processor,
        'top_tfidf': top_tfidf,
        'topics': topics,
        'similarity_matrix': similarity_matrix,
        'features': features_df
    }

# Run demonstration
# results = demonstrate_advanced_nlp()`
        }
      ]
    },
    {
      id: 'text-classification',
      title: 'Text Classification & Machine Learning',
      icon: '🎯',
      description: 'Build machine learning models for text classification and document analysis',
      content: `
        Text classification assigns predefined categories to text documents.
        Learn to build and evaluate ML models using traditional algorithms and modern deep learning approaches.
      `,
      keyTopics: [
        'Feature Engineering for Text',
        'Naive Bayes Classification',
        'Support Vector Machines (SVM)',
        'Logistic Regression for Text',
        'Random Forest and Ensemble Methods',
        'Model Evaluation and Metrics',
        'Cross-validation for Text Data',
        'Handling Imbalanced Text Datasets'
      ],
      codeExamples: [
        {
          title: 'Complete Text Classification Pipeline',
          description: 'Build end-to-end text classification system with multiple algorithms',
          code: `from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TextClassificationPipeline:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.pipelines = {}
        self.results = {}
        
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """Prepare train-test split"""
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Class distribution in training set:")
        print(pd.Series(y_train).value_counts())
        
        return X_train, X_test, y_train, y_test
    
    def create_pipelines(self):
        """Create ML pipelines with different algorithms"""
        # Naive Bayes with TF-IDF
        self.pipelines['nb_tfidf'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # SVM with TF-IDF
        self.pipelines['svm_tfidf'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', SVC(kernel='linear', C=1.0, random_state=42))
        ])
        
        # Logistic Regression with TF-IDF
        self.pipelines['lr_tfidf'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Random Forest with Count Vectorizer
        self.pipelines['rf_count'] = Pipeline([
            ('count', CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Ensemble approach - Logistic Regression with more features
        self.pipelines['lr_enhanced'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), 
                                    stop_words='english', sublinear_tf=True)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000, C=0.5))
        ])
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train all models and evaluate performance"""
        self.create_pipelines()
        
        for name, pipeline in self.pipelines.items():
            print(f"\\nTraining {name}...")
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = None
            
            # Get prediction probabilities if available
            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                y_pred_proba = pipeline.named_steps['classifier'].predict_proba(
                    pipeline.named_steps[list(pipeline.named_steps.keys())[0]].transform(X_test)
                )
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"Accuracy: {accuracy:.4f}")
        
        return self.results
    
    def cross_validate_models(self, X_train, y_train, cv=5):
        """Perform cross-validation for all models"""
        cv_results = {}
        
        for name, pipeline in self.pipelines.items():
            print(f"\\nCross-validating {name}...")
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
            
            cv_results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='lr_tfidf'):
        """Perform hyperparameter tuning for a specific model"""
        if model_name == 'lr_tfidf':
            param_grid = {
                'tfidf__max_features': [3000, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'classifier__C': [0.1, 0.5, 1.0, 2.0]
            }
        elif model_name == 'svm_tfidf':
            param_grid = {
                'tfidf__max_features': [3000, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier__C': [0.5, 1.0, 2.0]
            }
        elif model_name == 'nb_tfidf':
            param_grid = {
                'tfidf__max_features': [3000, 5000, 10000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'classifier__alpha': [0.01, 0.1, 1.0]
            }
        else:
            print(f"Hyperparameter tuning not configured for {model_name}")
            return None
        
        pipeline = self.pipelines[model_name]
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        print(f"\\nPerforming hyperparameter tuning for {model_name}...")
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def analyze_feature_importance(self, model_name='lr_tfidf', top_k=20):
        """Analyze feature importance for linear models"""
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        pipeline = self.results[model_name]['pipeline']
        classifier = pipeline.named_steps['classifier']
        vectorizer = pipeline.named_steps[list(pipeline.named_steps.keys())[0]]
        
        if hasattr(classifier, 'coef_'):
            feature_names = vectorizer.get_feature_names_out()
            coefficients = classifier.coef_
            
            if len(coefficients.shape) > 1:
                # Multi-class classification
                classes = classifier.classes_
                
                for i, class_name in enumerate(classes):
                    print(f"\\nTop features for class '{class_name}':")
                    
                    # Get coefficients for this class
                    class_coef = coefficients[i]
                    
                    # Get top positive features
                    top_positive_indices = class_coef.argsort()[-top_k//2:][::-1]
                    print("Most indicative features:")
                    for idx in top_positive_indices:
                        print(f"  {feature_names[idx]}: {class_coef[idx]:.4f}")
                    
                    # Get top negative features
                    top_negative_indices = class_coef.argsort()[:top_k//2]
                    print("Least indicative features:")
                    for idx in top_negative_indices:
                        print(f"  {feature_names[idx]}: {class_coef[idx]:.4f}")
            else:
                # Binary classification
                feature_importance = list(zip(feature_names, coefficients[0]))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print(f"\\nTop {top_k} most important features:")
                for feature, importance in feature_importance[:top_k]:
                    print(f"  {feature}: {importance:.4f}")
        else:
            print(f"Feature importance not available for {model_name}")
    
    def visualize_results(self, y_test):
        """Create comprehensive visualizations of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy comparison
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        ax1 = axes[0, 0]
        bars = ax1.bar(model_names, accuracies, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        
        # 2. Confusion matrix for best model
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_name, best_result = best_model
        
        ax2 = axes[0, 1]
        sns.heatmap(best_result['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', ax=ax2)
        ax2.set_title(f'Confusion Matrix - {best_name}')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # 3. Precision, Recall, F1 comparison
        ax3 = axes[1, 0]
        
        metrics_data = {'Model': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
        for name, result in self.results.items():
            report = result['classification_report']
            # Use weighted average for overall metrics
            metrics_data['Model'].append(name)
            metrics_data['Precision'].append(report['weighted avg']['precision'])
            metrics_data['Recall'].append(report['weighted avg']['recall'])
            metrics_data['F1-Score'].append(report['weighted avg']['f1-score'])
        
        x = np.arange(len(metrics_data['Model']))
        width = 0.25
        
        ax3.bar(x - width, metrics_data['Precision'], width, label='Precision', alpha=0.8)
        ax3.bar(x, metrics_data['Recall'], width, label='Recall', alpha=0.8)
        ax3.bar(x + width, metrics_data['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_title('Precision, Recall, and F1-Score Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics_data['Model'], rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # 4. Class-wise performance for best model
        ax4 = axes[1, 1]
        
        report = best_result['classification_report']
        classes = [key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
        
        if len(classes) <= 10:  # Only show if not too many classes
            class_f1_scores = [report[cls]['f1-score'] for cls in classes]
            
            ax4.bar(classes, class_f1_scores, color='lightgreen')
            ax4.set_title(f'F1-Score by Class - {best_name}')
            ax4.set_xlabel('Classes')
            ax4.set_ylabel('F1-Score')
            ax4.set_ylim(0, 1)
            plt.setp(ax4.get_xticklabels(), rotation=45)
        else:
            ax4.text(0.5, 0.5, f'Too many classes ({len(classes)}) to display', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Class Performance')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def predict_new_text(self, text, model_name=None):
        """Make predictions on new text"""
        if model_name is None:
            # Use the best performing model
            best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            model_name = best_model[0]
        
        if model_name not in self.results:
            print(f"Model {model_name} not found")
            return None
        
        pipeline = self.results[model_name]['pipeline']
        
        # Make prediction
        prediction = pipeline.predict([text])[0]
        
        # Get prediction probabilities if available
        if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
            vectorizer = pipeline.named_steps[list(pipeline.named_steps.keys())[0]]
            probabilities = pipeline.named_steps['classifier'].predict_proba(
                vectorizer.transform([text])
            )[0]
            
            classes = pipeline.named_steps['classifier'].classes_
            prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
            
            print(f"Predicted class: {prediction}")
            print("Class probabilities:")
            for cls, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {prob:.4f}")
            
            return prediction, prob_dict
        else:
            print(f"Predicted class: {prediction}")
            return prediction, None

# Demonstration function
def demonstrate_text_classification():
    # Sample dataset - news articles classification
    texts = [
        "The stock market reached new highs today as technology companies reported strong earnings.",
        "Scientists have discovered a new exoplanet that could potentially support life.",
        "The championship game was decided in overtime with a spectacular goal by the home team.",
        "New breakthrough in machine learning could revolutionize artificial intelligence research.",
        "Economic indicators suggest continued growth despite global uncertainties.",
        "Researchers publish findings on climate change impact on marine ecosystems.",
        "Baseball season opens with record attendance and exciting matchups.",
        "Deep learning models achieve human-level performance in medical diagnosis.",
        "Federal Reserve announces interest rate decision affecting mortgage markets.",
        "Space telescope captures stunning images of distant galaxy formation.",
        "Football draft brings new talent to struggling teams across the league.",
        "Natural language processing advances enable better human-computer interaction.",
        "Cryptocurrency prices fluctuate as regulatory concerns mount worldwide.",
        "Marine biologists discover new species in unexplored ocean depths",
        "Basketball playoffs feature intense competition and record-breaking performances",
        "Computer vision systems now surpass human accuracy in object recognition tasks"
    ]
    
    labels = [
        "business", "science", "sports", "technology",
        "business", "science", "sports", "technology",
        "business", "science", "sports", "technology",
        "business", "science", "sports", "technology"
    ]
    
    # Initialize classification pipeline
    classifier = TextClassificationPipeline()
    
    print("=== Text Classification Pipeline Demonstration ===\\n")
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(texts, labels, test_size=0.3)
    
    # Train and evaluate models
    results = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Cross-validation
    print("\\n" + "="*50)
    print("Cross-Validation Results:")
    cv_results = classifier.cross_validate_models(X_train, y_train, cv=3)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\\nBest performing model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    # Detailed classification report for best model
    print(f"\\nDetailed Classification Report for {best_model[0]}:")
    print(classification_report(y_test, best_model[1]['predictions']))
    
    # Feature importance analysis
    classifier.analyze_feature_importance(best_model[0], top_k=10)
    
    # Test with new examples
    print("\\n" + "="*50)
    print("Testing with new examples:")
    
    new_texts = [
        "Apple stock soars after quarterly earnings beat expectations",
        "NASA announces new mission to Mars with advanced rover technology",
        "Tennis championship final draws record television audience worldwide"
    ]
    
    for text in new_texts:
        print(f"\\nText: {text}")
        prediction, probabilities = classifier.predict_new_text(text, best_model[0])
    
    # Visualize results
    # classifier.visualize_results(y_test)
    
    return classifier, results

# Run demonstration
# classifier, results = demonstrate_text_classification()`
        }
      ]
    },
    {
      id: 'transformers-nlp',
      title: 'Modern NLP with Transformers',
      icon: '🤖',
      description: 'Implement state-of-the-art NLP using BERT, GPT, and Transformer architectures',
      content: `
        Transformers have revolutionized NLP with attention mechanisms and pre-trained models.
        Learn to use BERT, GPT, and other transformer models for various NLP tasks.
      `,
      keyTopics: [
        'Transformer Architecture',
        'BERT for Text Classification',
        'GPT for Text Generation',
        'Fine-tuning Pre-trained Models',
        'Hugging Face Transformers',
        'Attention Mechanisms',
        'Transfer Learning in NLP',
        'Model Deployment and Optimization'
      ],
      codeExamples: [
        {
          title: 'BERT Implementation with Hugging Face',
          description: 'Use pre-trained BERT models for various NLP tasks',
          code: `from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class BERTClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def preprocess_text(self, texts, max_length=512):
        """Tokenize and encode texts"""
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return encoding
    
    def create_dataset(self, texts, labels=None):
        """Create custom dataset for training/inference"""
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                item = {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten()
                }
                
                if self.labels is not None:
                    item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                
                return item
        
        return TextDataset(texts, labels, self.tokenizer)
    
    def train_model(self, train_texts, train_labels, val_texts=None, val_labels=None, 
                   epochs=3, batch_size=16, learning_rate=2e-5):
        """Fine-tune BERT model"""
        
        # Create datasets
        train_dataset = self.create_dataset(train_texts, train_labels)
        val_dataset = None
        if val_texts is not None and val_labels is not None:
            val_dataset = self.create_dataset(val_texts, val_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./bert_results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir='./bert_logs',
            logging_steps=100,
            evaluation_strategy='epoch' if val_dataset else 'no',
            save_strategy='epoch',
            load_best_model_at_end=True if val_dataset else False
        )
        
        # Metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'accuracy': accuracy_score(labels, predictions)}
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics if val_dataset else None
        )
        
        # Train model
        print("Starting BERT fine-tuning...")
        trainer.train()
        
        return trainer
    
    def predict(self, texts, batch_size=16):
        """Make predictions on new texts"""
        self.model.eval()
        
        dataset = self.create_dataset(texts)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)
                
                predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return predictions, probabilities
    
    def analyze_attention(self, text, layer=-1, head=0):
        """Visualize attention patterns"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        
        # Get attention weights for specified layer and head
        attention = attentions[layer][0, head].cpu().numpy()
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return attention, tokens

class TransformerTextGenerator:
    def __init__(self, model_name='gpt2'):
        self.model_name = model_name
        self.generator = pipeline('text-generation', model=model_name)
        
    def generate_text(self, prompt, max_length=100, num_return_sequences=1, 
                     temperature=0.7, top_p=0.9):
        """Generate text using GPT-2"""
        results = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.generator.tokenizer.eos_token_id
        )
        
        return [result['generated_text'] for result in results]
    
    def complete_text(self, prompt, max_new_tokens=50):
        """Complete text with controlled generation"""
        inputs = self.generator.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.generator.model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
        
        generated_text = self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

class MultiTaskNLP:
    def __init__(self):
        # Initialize different pipelines for various tasks
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.ner_extractor = pipeline('ner', aggregation_strategy='simple')
        self.question_answerer = pipeline('question-answering')
        self.summarizer = pipeline('summarization')
        self.translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-es')
        
    def analyze_sentiment(self, texts):
        """Perform sentiment analysis"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = self.sentiment_analyzer(texts)
        return results
    
    def extract_entities(self, text):
        """Extract named entities"""
        entities = self.ner_extractor(text)
        return entities
    
    def answer_question(self, context, question):
        """Answer questions based on context"""
        result = self.question_answerer(question=question, context=context)
        return result
    
    def summarize_text(self, text, max_length=100, min_length=30):
        """Summarize long text"""
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    
    def translate_text(self, text, target_lang='es'):
        """Translate text (English to Spanish by default)"""
        if target_lang == 'es':
            result = self.translator(text)
            return result[0]['translation_text']
        else:
            print(f"Translation to {target_lang} not configured")
            return None

# Comprehensive demonstration
def demonstrate_transformers():
    print("=== Modern NLP with Transformers Demonstration ===\\n")
    
    # Sample data for demonstration
    sample_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst service I've ever experienced. Terrible!",
        "The weather is okay today, nothing special.",
        "Outstanding quality and excellent customer support. Highly recommended!",
        "Not satisfied with the purchase. Could be better."
    ]
    
    sample_labels = [1, 0, 2, 1, 0]  # 0: negative, 1: positive, 2: neutral
    
    # 1. BERT Classification
    print("1. BERT Text Classification:")
    bert_classifier = BERTClassifier(model_name='distilbert-base-uncased', num_labels=3)
    
    # For demonstration, we'll use a smaller subset
    # In practice, you'd need more data for training
    print("BERT classifier initialized (training would require larger dataset)")
    
    # Make predictions with pre-trained model (sentiment analysis)
    predictions, probabilities = bert_classifier.predict(sample_texts[:3])
    for i, (text, pred, prob) in enumerate(zip(sample_texts[:3], predictions, probabilities)):
        print(f"Text: {text[:50]}...")
        print(f"Prediction: {pred}, Probabilities: {prob}\\n")
    
    # 2. Text Generation with GPT-2
    print("2. Text Generation with GPT-2:")
    text_generator = TransformerTextGenerator('gpt2')
    
    prompts = [
        "The future of artificial intelligence is",
        "In the world of data science,",
        "Machine learning algorithms can"
    ]
    
    for prompt in prompts:
        generated = text_generator.generate_text(prompt, max_length=80, num_return_sequences=1)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated[0]}\\n")
    
    # 3. Multi-task NLP
    print("3. Multi-task NLP Pipeline:")
    nlp_pipeline = MultiTaskNLP()
    
    # Sentiment analysis
    sentiments = nlp_pipeline.analyze_sentiment(sample_texts)
    print("Sentiment Analysis:")
    for text, sentiment in zip(sample_texts, sentiments):
        print(f"Text: {text[:40]}...")
        print(f"Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.3f})\\n")
    
    # Named entity recognition
    sample_text_ner = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
    entities = nlp_pipeline.extract_entities(sample_text_ner)
    print(f"Named Entity Recognition for: {sample_text_ner}")
    for entity in entities:
        print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.3f}")
    print()
    
    # Question answering
    context = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."
    question = "What is machine learning?"
    answer = nlp_pipeline.answer_question(context, question)
    print("Question Answering:")
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {answer['answer']} (confidence: {answer['score']:.3f})\\n")
    
    # Text summarization
    long_text = """
    Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language, in particular how to program computers 
    to process and analyze large amounts of natural language data. The result is a computer capable of understanding 
    the contents of documents, including the contextual nuances of the language within them. The technology can then 
    accurately extract information and insights contained in the documents as well as categorize and organize the 
    documents themselves. Challenges in natural language processing frequently involve speech recognition, natural 
    language understanding, and natural language generation.
    """
    
    summary = nlp_pipeline.summarize_text(long_text, max_length=60, min_length=20)
    print("Text Summarization:")
    print(f"Original text length: {len(long_text)} characters")
    print(f"Summary: {summary}")
    print(f"Summary length: {len(summary)} characters\\n")
    
    # 4. Advanced analysis
    print("4. Advanced Analysis:")
    
    # Compare different models' performance on same task
    models_comparison = {
        'distilbert-base-uncased': 'DistilBERT (lightweight)',
        'roberta-base': 'RoBERTa (robust)',
        'albert-base-v2': 'ALBERT (parameter sharing)'
    }
    
    print("Model Comparison for Sentiment Analysis:")
    test_text = "This transformer-based approach is revolutionary and will change everything!"
    
    for model_name, description in models_comparison.items():
        try:
            classifier = pipeline('sentiment-analysis', model=model_name)
            result = classifier(test_text)
            print(f"{description}: {result[0]['label']} (score: {result[0]['score']:.3f})")
        except Exception as e:
            print(f"{description}: Error loading model - {str(e)[:50]}")
    
    print("\\nTransformers demonstration completed!")
    
    return {
        'bert_classifier': bert_classifier,
        'text_generator': text_generator,
        'nlp_pipeline': nlp_pipeline,
        'sample_results': {
            'sentiments': sentiments,
            'entities': entities,
            'qa_answer': answer,
            'summary': summary
        }
    }

# Run demonstration
# results = demonstrate_transformers()`
        }
      ]
    }
  ]

  return (
    <div className="page">
      <div className="content">
        <div className="page-header">
          <h1>📝 Complete Natural Language Processing</h1>
          <p className="page-description">
            Master NLP from fundamentals to advanced transformers. Learn text processing, 
            machine learning for text, and modern transformer architectures like BERT and GPT.
          </p>
        </div>

        <div className="learning-path">
          <h2>🗺️ NLP Learning Path</h2>
          <div className="path-steps">
            <div className="path-step">
              <div className="step-number">1</div>
              <h3>NLP Fundamentals</h3>
              <p>Master text preprocessing, tokenization, and linguistic analysis with NLTK and spaCy</p>
            </div>
            <div className="path-step">
              <div className="step-number">2</div>
              <h3>Text Classification</h3>
              <p>Build ML models for text classification using traditional and modern approaches</p>
            </div>
            <div className="path-step">
              <div className="step-number">3</div>
              <h3>Advanced Processing</h3>
              <p>Implement TF-IDF, topic modeling, and advanced feature engineering techniques</p>
            </div>
            <div className="path-step">
              <div className="step-number">4</div>
              <h3>Transformers & BERT</h3>
              <p>Use state-of-the-art transformer models for various NLP tasks</p>
            </div>
          </div>
        </div>

        <div className="section-tabs">
          {sections.map((section, index) => (
            <button
              key={section.id}
              className={`tab-button ${activeSection === index ? 'active' : ''}`}
              onClick={() => setActiveSection(index)}
            >
              <span className="tab-icon">{section.icon}</span>
              {section.title}
            </button>
          ))}
        </div>

        <div className="section-content">
          {sections.map((section, index) => (
            <div
              key={section.id}
              className={`section ${activeSection === index ? 'active' : ''}`}
            >
              <div className="section-header">
                <h2>
                  <span className="section-icon">{section.icon}</span>
                  {section.title}
                </h2>
                <p className="section-description">{section.description}</p>
              </div>

              <div className="section-overview">
                <p>{section.content}</p>
              </div>

              <div className="key-topics">
                <h3>🎯 Key Topics Covered</h3>
                <div className="topics-grid">
                  {section.keyTopics.map((topic, idx) => (
                    <div key={idx} className="topic-item">
                      <span className="topic-bullet">▶</span>
                      {topic}
                    </div>
                  ))}
                </div>
              </div>

              <div className="code-examples">
                <h3>💻 Code Examples & Implementation</h3>
                {section.codeExamples.map((example, idx) => (
                  <div key={idx} className="code-example">
                    <div className="example-header">
                      <h4>{example.title}</h4>
                      <p>{example.description}</p>
                      <button
                        className="toggle-code"
                        onClick={() => toggleCode(section.id, idx)}
                      >
                        {expandedCode[`${section.id}-${idx}`] ? 'Hide Code' : 'Show Code'}
                      </button>
                    </div>
                    
                    {expandedCode[`${section.id}-${idx}`] && (
                      <div className="code-block">
                        <pre><code>{example.code}</code></pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="practice-exercises">
                <h3>🏋️ Practice Exercises</h3>
                <div className="exercises">
                  <div className="exercise">
                    <h4>Beginner Exercise</h4>
                    <p>Build a sentiment analysis tool using NLTK and traditional ML algorithms.</p>
                  </div>
                  <div className="exercise">
                    <h4>Intermediate Exercise</h4>
                    <p>Create a news article classifier with TF-IDF features and ensemble methods.</p>
                  </div>
                  <div className="exercise">
                    <h4>Advanced Exercise</h4>
                    <p>Fine-tune BERT for a specific domain classification task and deploy as API.</p>
                  </div>
                </div>
              </div>

              <div className="real-world-projects">
                <h3>🚀 Real-World Project Ideas</h3>
                <div className="projects-grid">
                  <div className="project-card">
                    <h4>Social Media Sentiment Monitor</h4>
                    <p>Build real-time sentiment analysis for social media posts with trend detection.</p>
                  </div>
                  <div className="project-card">
                    <h4>Document Summarization System</h4>
                    <p>Create automated summarization for legal documents, research papers, or news articles.</p>
                  </div>
                  <div className="project-card">
                    <h4>Intelligent Chatbot</h4>
                    <p>Develop a context-aware chatbot using transformers and conversation management.</p>
                  </div>
                  <div className="project-card">
                    <h4>Content Moderation Tool</h4>
                    <p>Build automated content moderation system for detecting harmful or inappropriate text.</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="next-steps">
          <h2>🎯 Next Steps in NLP</h2>
          <div className="next-steps-grid">
            <div className="next-step">
              <h3>🤖 Advanced AI</h3>
              <p>Explore GPT-3/4, ChatGPT, and large language model applications</p>
            </div>
            <div className="next-step">
              <h3>🌐 Multimodal AI</h3>
              <p>Combine NLP with computer vision for multimodal understanding</p>
            </div>
            <div className="next-step">
              <h3>🏭 Production Systems</h3>
              <p>Deploy NLP models at scale with proper monitoring and maintenance</p>
            </div>
            <div className="next-step">
              <h3>🔬 Specialized Domains</h3>
              <p>Apply NLP to healthcare, finance, legal, and other specialized fields</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default NaturalLanguageProcessingComplete