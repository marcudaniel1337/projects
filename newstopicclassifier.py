import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pickle
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data - yeah, this always trips people up on first run
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class NewsTopicClassifier:
    """
    A sophisticated news topic classifier that actually works in the real world.
    
    This isn't just another tutorial classifier - it's built to handle the messy,
    inconsistent nature of real news data. I've seen too many classifiers that
    work great on clean datasets but fall apart when you throw actual scraped
    news articles at them.
    """
    
    def __init__(self, random_state=42):
        # Set random state for reproducibility - trust me, you'll thank me later
        # when you're trying to debug why your model performance keeps changing
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        
        # These are the topic categories we'll classify into
        # I chose these based on what I see most commonly in news feeds
        self.categories = [
            'politics', 'technology', 'sports', 'business', 'entertainment',
            'health', 'science', 'world_news', 'crime', 'environment'
        ]
        
        # Stop words that are actually useful to remove for news classification
        # The default NLTK list is okay, but we need some news-specific ones
        self.custom_stopwords = set(stopwords.words('english')).union({
            'said', 'says', 'according', 'report', 'reported', 'reports',
            'news', 'article', 'story', 'today', 'yesterday', 'tomorrow',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
        })
    
    def preprocess_text(self, text):
        """
        Clean and preprocess the text data.
        
        This is where the magic happens - or where everything goes wrong if you're not careful.
        News text is particularly messy because it comes from so many different sources
        with different formatting, encoding issues, and writing styles.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase - seems obvious but you'd be surprised how often this is forgotten
        text = text.lower()
        
        # Remove URLs - news articles are full of these and they just add noise
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses - same reasoning as URLs
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace and newlines - news scraping often introduces weird spacing
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-alphabetic characters but keep spaces
        # I'm being a bit aggressive here, but numbers and punctuation rarely help with topic classification
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize the text - split it into individual words
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        # Lemmatization is better than stemming for this use case because we want real words
        processed_tokens = []
        for token in tokens:
            if token not in self.custom_stopwords and len(token) > 2:  # Skip very short words
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def create_sample_data(self, n_samples=1000):
        """
        Generate sample news data for demonstration.
        
        In a real scenario, you'd load this from your actual news database or API.
        I'm creating realistic-looking fake data that covers our topic categories.
        The key is making sure each category has distinctive vocabulary patterns.
        """
        np.random.seed(self.random_state)
        
        # Sample texts for each category - these are designed to have clear distinguishing features
        sample_texts = {
            'politics': [
                "The senator announced new legislation regarding healthcare reform during today's press conference",
                "Election results show a tight race between the incumbent and challenger in the gubernatorial race",
                "Congress voted on the controversial bill that has divided both parties for months",
                "The president's approval ratings have fluctuated following recent policy announcements",
                "Local mayor proposes budget changes that would affect municipal services and taxation"
            ],
            'technology': [
                "Apple announces new iPhone model with enhanced camera capabilities and longer battery life",
                "Artificial intelligence breakthrough enables more accurate medical diagnosis predictions",
                "Cybersecurity experts warn of new ransomware threats targeting small businesses",
                "Tesla reports record quarterly earnings driven by electric vehicle sales growth",
                "Google's latest algorithm update affects search rankings for millions of websites"
            ],
            'sports': [
                "The championship game ended with a dramatic overtime victory for the home team",
                "Star quarterback suffers injury that may sideline him for the remainder of the season",
                "Olympic preparations continue as athletes train for upcoming international competition",
                "Trade deadline approaches with several teams looking to strengthen their rosters",
                "Coach announces retirement after successful career spanning three decades"
            ],
            'business': [
                "Stock market reaches new highs as investors show confidence in economic recovery",
                "Major corporation announces layoffs affecting thousands of employees nationwide",
                "Startup secures significant funding round led by prominent venture capital firm",
                "Oil prices fluctuate amid global supply chain disruptions and geopolitical tensions",
                "Retail sales figures exceed expectations during holiday shopping season"
            ],
            'entertainment': [
                "Hollywood blockbuster breaks box office records in opening weekend performance",
                "Celebrity couple announces engagement during red carpet interview at film premiere",
                "Streaming service reveals most-watched shows and movies from the past year",
                "Music festival lineup announced featuring headliners from multiple genres",
                "Award ceremony celebrates outstanding achievements in television and film"
            ],
            'health': [
                "Medical researchers publish findings on new treatment for rare genetic disorder",
                "Health officials recommend updated vaccination schedules for children and adults",
                "Mental health awareness campaign launches to address rising anxiety and depression rates",
                "Clinical trial results show promising outcomes for experimental cancer therapy",
                "Nutritionists advise on dietary changes to improve cardiovascular health outcomes"
            ],
            'science': [
                "Astronomers discover potentially habitable exoplanet using advanced telescope technology",
                "Climate scientists publish research on accelerating ice sheet melting patterns",
                "Breakthrough in quantum computing brings practical applications closer to reality",
                "Archaeological team uncovers ancient artifacts providing insights into historical civilization",
                "Marine biologists document new species discovered in deep ocean exploration mission"
            ],
            'world_news': [
                "International summit addresses global trade agreements and economic cooperation",
                "Natural disaster relief efforts continue in affected regions with humanitarian aid",
                "Diplomatic negotiations progress toward resolving long-standing territorial disputes",
                "Cultural exchange program promotes understanding between different nations and peoples",
                "Immigration policies face scrutiny as countries balance security and humanitarian concerns"
            ],
            'crime': [
                "Police investigation leads to arrests in connection with organized crime syndicate",
                "Court proceedings begin for high-profile fraud case involving millions of dollars",
                "Detective work solves cold case that has puzzled investigators for over a decade",
                "Security camera footage provides crucial evidence in robbery investigation",
                "Community safety initiative aims to reduce violent crime rates in urban areas"
            ],
            'environment': [
                "Renewable energy project generates clean electricity for thousands of households",
                "Conservation efforts protect endangered species habitat from development pressures",
                "Pollution levels in major city exceed safe standards prompting health warnings",
                "Climate change impacts become visible in shifting weather patterns and temperatures",
                "Environmental activist group organizes cleanup campaign for local waterways"
            ]
        }
        
        # Generate the dataset by sampling from our templates
        data = []
        samples_per_category = n_samples // len(self.categories)
        
        for category, texts in sample_texts.items():
            for _ in range(samples_per_category):
                # Pick a random template and add some variation
                base_text = np.random.choice(texts)
                # In real life, you might add more sophisticated text generation here
                data.append({'text': base_text, 'category': category})
        
        return pd.DataFrame(data)
    
    def setup_model_pipeline(self):
        """
        Create the machine learning pipeline.
        
        I'm using an ensemble approach here because news classification can be tricky.
        Different algorithms excel at different aspects - some are great with common topics,
        others handle edge cases better. Combining them usually gives more robust results.
        """
        
        # TF-IDF vectorizer with carefully chosen parameters
        # These settings work well for news text based on my experimentation
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit vocabulary size to prevent overfitting
            ngram_range=(1, 2),  # Include both single words and word pairs
            min_df=2,  # Ignore words that appear in fewer than 2 documents
            max_df=0.95,  # Ignore words that appear in more than 95% of documents
            stop_words=None  # We handle stopwords in preprocessing
        )
        
        # Individual classifiers that we'll combine
        # Each brings different strengths to the ensemble
        rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=self.random_state,
            n_jobs=-1  # Use all available CPU cores
        )
        
        svm_classifier = SVC(
            kernel='linear',  # Linear kernel works well for text classification
            probability=True,  # We need probabilities for the voting classifier
            random_state=self.random_state
        )
        
        nb_classifier = MultinomialNB(
            alpha=0.1  # Smoothing parameter - tuned through experimentation
        )
        
        lr_classifier = LogisticRegression(
            max_iter=1000,  # Increase iterations to ensure convergence
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Voting classifier combines predictions from all models
        # 'soft' voting uses predicted probabilities rather than just class predictions
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf_classifier),
                ('svm', svm_classifier),
                ('nb', nb_classifier),
                ('lr', lr_classifier)
            ],
            voting='soft'  # Use probability-based voting
        )
    
    def train(self, df):
        """
        Train the classifier on the provided dataset.
        
        This is where we actually build the model. The preprocessing and feature extraction
        happen here, followed by the actual machine learning training.
        """
        print("Starting training process...")
        print(f"Dataset size: {len(df)} articles")
        print(f"Categories: {df['category'].value_counts().to_dict()}")
        
        # Preprocess all the text data
        print("Preprocessing text data...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove any empty texts that might have resulted from preprocessing
        df = df[df['processed_text'].str.len() > 0]
        print(f"After preprocessing: {len(df)} articles remaining")
        
        # Split the data for training and testing
        # 80/20 split is pretty standard, but you might want to adjust based on your dataset size
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['category'],
            test_size=0.2,
            random_state=self.random_state,
            stratify=df['category']  # Ensure balanced splits across categories
        )
        
        # Set up the model pipeline
        self.setup_model_pipeline()
        
        # Transform text to numerical features
        print("Creating TF-IDF features...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Train the ensemble model
        print("Training ensemble model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate the model performance
        print("Evaluating model performance...")
        y_pred = self.model.predict(X_test_tfidf)
        
        # Print detailed performance metrics
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store test data for potential further analysis
        self.X_test = X_test_tfidf
        self.y_test = y_test
        self.y_pred = y_pred
        
        return self
    
    def predict(self, texts):
        """
        Predict categories for new texts.
        
        This is the function you'll use most often once the model is trained.
        It handles both single texts and lists of texts.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Handle both single strings and lists
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess the input texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform to TF-IDF features
        text_tfidf = self.vectorizer.transform(processed_texts)
        
        # Make predictions
        predictions = self.model.predict(text_tfidf)
        probabilities = self.model.predict_proba(text_tfidf)
        
        # Return results with confidence scores
        results = []
        for i, pred in enumerate(predictions):
            # Get the confidence score for the predicted class
            pred_idx = list(self.model.classes_).index(pred)
            confidence = probabilities[i][pred_idx]
            
            results.append({
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'predicted_category': pred,
                'confidence': confidence,
                'all_probabilities': dict(zip(self.model.classes_, probabilities[i]))
            })
        
        return results if len(results) > 1 else results[0]
    
    def get_feature_importance(self, top_n=10):
        """
        Get the most important features for each category.
        
        This is really useful for understanding what the model is actually learning.
        Sometimes you discover that your model is picking up on weird artifacts
        rather than meaningful content.
        """
        if self.model is None:
            raise ValueError("Model hasn't been trained yet.")
        
        # Get feature names from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        
        # For ensemble models, we'll use the logistic regression component
        # since it's most interpretable
        lr_model = self.model.named_estimators_['lr']
        
        # Get coefficients for each class
        importance_dict = {}
        for i, category in enumerate(lr_model.classes_):
            # Get the coefficients for this class
            coefficients = lr_model.coef_[i]
            
            # Get indices of top features
            top_indices = np.argsort(np.abs(coefficients))[::-1][:top_n]
            
            # Create list of (feature, coefficient) tuples
            top_features = [(feature_names[idx], coefficients[idx]) for idx in top_indices]
            importance_dict[category] = top_features
        
        return importance_dict
    
    def plot_confusion_matrix(self):
        """
        Create a visualization of the confusion matrix.
        
        This helps you see which categories the model confuses with each other.
        Very useful for identifying areas where you might need more training data
        or better feature engineering.
        """
        if not hasattr(self, 'y_test'):
            raise ValueError("No test data available. Train the model first.")
        
        # Create confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Plot with seaborn for better aesthetics
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.model.classes_,
                   yticklabels=self.model.classes_)
        plt.title('News Topic Classification - Confusion Matrix')
        plt.xlabel('Predicted Category')
        plt.ylabel('Actual Category')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        You definitely want to do this after training - nobody wants to retrain
        their model every time they restart their application.
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'categories': self.categories,
            'custom_stopwords': self.custom_stopwords
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a previously trained model from disk.
        
        This lets you skip the training step if you already have a good model.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.categories = model_data['categories']
        self.custom_stopwords = model_data['custom_stopwords']
        
        print(f"Model loaded from {filepath}")
        return self


# Example usage and demonstration
if __name__ == "__main__":
    print("=== News Topic Classifier Demo ===\n")
    
    # Initialize the classifier
    classifier = NewsTopicClassifier()
    
    # Create sample data - in real life, you'd load your actual news data here
    print("Generating sample news data...")
    sample_data = classifier.create_sample_data(n_samples=2000)
    
    # Train the model
    print("\nTraining the classifier...")
    classifier.train(sample_data)
    
    # Test with some example articles
    test_articles = [
        "The Federal Reserve announced an interest rate increase to combat rising inflation pressures",
        "Scientists discover breakthrough treatment for Alzheimer's disease in clinical trials",
        "Championship basketball game goes into triple overtime with record-breaking attendance",
        "New smartphone features artificial intelligence for improved photo processing capabilities",
        "Environmental activists protest against proposed pipeline construction through protected wilderness"
    ]
    
    print("\n=== Testing with Example Articles ===")
    for article in test_articles:
        result = classifier.predict(article)
        print(f"\nArticle: {result['text']}")
        print(f"Predicted Category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    # Show feature importance
    print("\n=== Most Important Features by Category ===")
    importance = classifier.get_feature_importance(top_n=5)
    for category, features in importance.items():
        print(f"\n{category.upper()}:")
        for feature, score in features:
            print(f"  {feature}: {score:.3f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    classifier.plot_confusion_matrix()
    
    # Save the model for future use
    print("\nSaving model...")
    classifier.save_model('news_classifier_model.pkl')
    
    print("\n=== Demo Complete ===")
    print("The classifier is now ready for use!")
    print("You can load it later with: classifier.load_model('news_classifier_model.pkl')")
