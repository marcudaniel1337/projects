import numpy as np
import pandas as pd
import re
import string
from collections import Counter, defaultdict
from math import log, sqrt, exp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from abc import ABC, abstractmethod

class TextPreprocessor:
    """
    Handles all the messy text cleaning that's crucial for spam detection
    
    Think of this as the 'janitor' of our system - it cleans up the text
    so our algorithms can focus on the important patterns rather than
    getting distracted by inconsistent formatting, typos, etc.
    """
    
    def __init__(self):
        # Common spam patterns we'll want to detect
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        self.phone_pattern = re.compile(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}')
        self.money_pattern = re.compile(r'\$[0-9,]+(?:\.[0-9]{2})?|[0-9,]+\s*(?:dollars?|usd|\$)')
        
        # Spam-specific words that are strong indicators
        self.spam_indicators = {
            'urgent', 'free', 'winner', 'congratulations', 'claim', 'prize',
            'money', 'cash', 'loan', 'credit', 'debt', 'mortgage', 'investment',
            'viagra', 'pharmacy', 'pills', 'medicine', 'health', 'weight',
            'click', 'subscribe', 'unsubscribe', 'remove', 'stop',
            'guarantee', 'risk-free', 'limited', 'offer', 'deal', 'discount'
        }
    
    def clean_text(self, text):
        """
        Main text cleaning pipeline - this is where we normalize everything
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Handle common spam tricks like l3tt3r substitution
        text = self._normalize_leetspeak(text)
        
        # Remove excessive punctuation (like "!!!!!!" -> "!")
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def _normalize_leetspeak(self, text):
        """
        Convert common leetspeak substitutions back to normal letters
        Spammers often use these to bypass simple filters
        """
        replacements = {
            '3': 'e', '4': 'a', '5': 's', '7': 't', '0': 'o',
            '1': 'i', '@': 'a', '$': 's'
        }
        
        for leet, normal in replacements.items():
            text = text.replace(leet, normal)
        
        return text
    
    def extract_features(self, text):
        """
        Extract various features that help distinguish spam from ham
        This is the 'detective work' - looking for patterns that spammers use
        """
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Suspicious formatting patterns
        features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        features['punct_ratio'] = sum(1 for c in text if c in string.punctuation) / max(len(text), 1)
        
        # Spam-specific patterns
        features['url_count'] = len(self.url_pattern.findall(text))
        features['email_count'] = len(self.email_pattern.findall(text))
        features['phone_count'] = len(self.phone_pattern.findall(text))
        features['money_mentions'] = len(self.money_pattern.findall(text))
        
        # Spam keyword density
        words = text.lower().split()
        spam_word_count = sum(1 for word in words if word in self.spam_indicators)
        features['spam_word_ratio'] = spam_word_count / max(len(words), 1)
        
        # Repetition patterns (spammers often repeat words for emphasis)
        word_counts = Counter(words)
        if word_counts:
            features['max_word_freq'] = max(word_counts.values())
            features['unique_word_ratio'] = len(word_counts) / len(words)
        else:
            features['max_word_freq'] = 0
            features['unique_word_ratio'] = 0
        
        return features

class TFIDFVectorizer:
    """
    Term Frequency-Inverse Document Frequency vectorizer
    
    This is the heart of text analysis - it converts text into numbers
    that represent how important each word is. Think of it as creating
    a 'fingerprint' for each document based on its word usage patterns.
    """
    
    def __init__(self, max_features=1000, min_df=2, max_df=0.8):
        self.max_features = max_features  # Only keep top N most important words
        self.min_df = min_df             # Ignore words that appear in < N documents
        self.max_df = max_df             # Ignore words that appear in > N% of documents
        
        self.vocabulary = {}
        self.idf_values = {}
        self.feature_names = []
    
    def _tokenize(self, text):
        """
        Break text into individual words/tokens
        We're being smart about this - removing punctuation but keeping meaningful parts
        """
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        
        # Remove very short tokens (usually not meaningful)
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def fit(self, documents):
        """
        Learn the vocabulary and IDF values from the training documents
        This is where we figure out which words are most informative
        """
        # Step 1: Count how many documents each word appears in
        doc_freq = defaultdict(int)
        total_docs = len(documents)
        
        print(f"Processing {total_docs} documents to build vocabulary...")
        
        for doc in documents:
            tokens = set(self._tokenize(doc))  # Use set to count each word only once per doc
            for token in tokens:
                doc_freq[token] += 1
        
        # Step 2: Filter words based on document frequency
        filtered_words = []
        for word, freq in doc_freq.items():
            # Skip words that appear too rarely or too frequently
            if freq >= self.min_df and freq / total_docs <= self.max_df:
                filtered_words.append((word, freq))
        
        # Step 3: Keep only the most common words (up to max_features)
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        self.feature_names = [word for word, _ in filtered_words[:self.max_features]]
        
        # Step 4: Create vocabulary mapping and calculate IDF values
        self.vocabulary = {word: idx for idx, word in enumerate(self.feature_names)}
        
        for word in self.feature_names:
            # IDF formula: log(total_docs / docs_containing_word)
            # Words that appear in fewer documents get higher IDF scores
            self.idf_values[word] = log(total_docs / doc_freq[word])
        
        print(f"Built vocabulary with {len(self.feature_names)} features")
    
    def transform(self, documents):
        """
        Convert documents to TF-IDF vectors
        Each document becomes a vector where each dimension represents a word's importance
        """
        if not self.vocabulary:
            raise ValueError("Must call fit() before transform()")
        
        vectors = []
        
        for doc in documents:
            # Count word frequencies in this document
            tokens = self._tokenize(doc)
            tf_counts = Counter(tokens)
            
            # Create TF-IDF vector for this document
            vector = np.zeros(len(self.feature_names))
            
            for word, tf in tf_counts.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    # TF-IDF = (term_freq / total_terms) * IDF
                    tf_normalized = tf / len(tokens)
                    vector[idx] = tf_normalized * self.idf_values[word]
            
            vectors.append(vector)
        
        return np.array(vectors)
    
    def fit_transform(self, documents):
        """Convenience method to fit and transform in one step"""
        self.fit(documents)
        return self.transform(documents)

class NaiveBayesClassifier:
    """
    Naive Bayes classifier - assumes features are independent
    
    This is a probabilistic classifier that's particularly good for text.
    It's 'naive' because it assumes words are independent (which isn't true),
    but it works surprisingly well in practice and is very fast.
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Smoothing parameter to handle unseen words
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []
    
    def fit(self, X, y):
        """
        Learn the probability distributions from training data
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        print(f"Training Naive Bayes on {n_samples} samples with {n_features} features...")
        
        # Calculate class priors P(class)
        for cls in self.classes:
            class_samples = np.sum(y == cls)
            self.class_priors[cls] = class_samples / n_samples
        
        # Calculate feature probabilities P(feature|class)
        for cls in self.classes:
            # Get all samples belonging to this class
            class_mask = (y == cls)
            class_features = X[class_mask]
            
            # Calculate mean and variance for each feature in this class
            # We'll use Gaussian assumption for continuous features
            feature_means = np.mean(class_features, axis=0)
            feature_vars = np.var(class_features, axis=0) + self.alpha  # Add smoothing
            
            self.feature_probs[cls] = {
                'means': feature_means,
                'vars': feature_vars
            }
    
    def _gaussian_prob(self, x, mean, var):
        """Calculate Gaussian probability density"""
        if var == 0:
            return 1.0 if x == mean else 1e-10
        
        exponent = -0.5 * ((x - mean) ** 2) / var
        return (1.0 / sqrt(2 * np.pi * var)) * exp(exponent)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for each sample
        """
        probabilities = []
        
        for sample in X:
            sample_probs = {}
            
            for cls in self.classes:
                # Start with class prior probability
                log_prob = log(self.class_priors[cls])
                
                # Add log probabilities for each feature
                means = self.feature_probs[cls]['means']
                vars = self.feature_probs[cls]['vars']
                
                for i, feature_val in enumerate(sample):
                    # Use log probabilities to avoid numerical underflow
                    prob = self._gaussian_prob(feature_val, means[i], vars[i])
                    log_prob += log(max(prob, 1e-10))  # Avoid log(0)
                
                sample_probs[cls] = log_prob
            
            # Convert back to probabilities and normalize
            max_log_prob = max(sample_probs.values())
            probs = {}
            total_prob = 0
            
            for cls in self.classes:
                probs[cls] = exp(sample_probs[cls] - max_log_prob)
                total_prob += probs[cls]
            
            # Normalize to get proper probabilities
            for cls in self.classes:
                probs[cls] /= total_prob
            
            probabilities.append([probs[cls] for cls in self.classes])
        
        return np.array(probabilities)
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]

class LogisticRegression:
    """
    Logistic Regression implemented from scratch
    
    This is a linear classifier that uses the logistic function to
    map any real value to a probability between 0 and 1. It's like
    drawing a line (or hyperplane) to separate spam from ham.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function - maps any real number to (0,1)
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _cost_function(self, y_true, y_pred):
        """
        Logistic regression cost function (cross-entropy)
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        print(f"Training Logistic Regression with {n_samples} samples, {n_features} features...")
        
        prev_cost = float('inf')
        
        for i in range(self.max_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            
            # Calculate cost
            cost = self._cost_function(y, y_pred)
            self.cost_history.append(cost)
            
            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
            prev_cost = cost
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost:.6f}")
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)

class SpamClassifier:
    """
    Main spam classifier that combines everything together
    
    This is the 'conductor' of our orchestra - it coordinates all the different
    components to create a complete spam detection system.
    """
    
    def __init__(self, algorithm='naive_bayes', tfidf_features=1000):
        self.algorithm = algorithm
        self.preprocessor = TextPreprocessor()
        self.tfidf_vectorizer = TFIDFVectorizer(max_features=tfidf_features)
        
        # Initialize the chosen algorithm
        if algorithm == 'naive_bayes':
            self.classifier = NaiveBayesClassifier()
        elif algorithm == 'logistic_regression':
            self.classifier = LogisticRegression()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.feature_names = []
        self.is_fitted = False
    
    def _extract_all_features(self, texts):
        """
        Extract both TF-IDF features and hand-crafted features
        """
        print("Extracting TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        print("Extracting hand-crafted features...")
        handcrafted_features = []
        for text in texts:
            features = self.preprocessor.extract_features(text)
            handcrafted_features.append(list(features.values()))
        
        handcrafted_features = np.array(handcrafted_features)
        
        # Combine TF-IDF and hand-crafted features
        combined_features = np.hstack([tfidf_features, handcrafted_features])
        
        return combined_features
    
    def fit(self, texts, labels):
        """
        Train the complete spam classifier
        """
        print(f"Training spam classifier using {self.algorithm}...")
        print(f"Dataset: {len(texts)} samples")
        
        # Clean texts
        print("Preprocessing texts...")
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer.fit(cleaned_texts)
        
        # Extract all features
        X = self._extract_all_features(cleaned_texts)
        
        # Convert labels to binary (0 for ham, 1 for spam)
        y = np.array([1 if label.lower() == 'spam' else 0 for label in labels])
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Train the classifier
        self.classifier.fit(X, y)
        self.is_fitted = True
        
        print("Training completed!")
    
    def predict(self, texts):
        """
        Predict whether texts are spam or ham
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess texts
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        # Extract features
        X = self._extract_all_features(cleaned_texts)
        
        # Make predictions
        predictions = self.classifier.predict(X)
        
        # Convert back to labels
        return ['spam' if pred == 1 else 'ham' for pred in predictions]
    
    def predict_proba(self, texts):
        """
        Predict spam probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess texts
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        # Extract features
        X = self._extract_all_features(cleaned_texts)
        
        # Get probabilities
        if hasattr(self.classifier, 'predict_proba'):
            probas = self.classifier.predict_proba(X)
            if probas.ndim == 2:
                return probas[:, 1]  # Return probability of spam class
            else:
                return probas
        else:
            # Fallback for classifiers without predict_proba
            predictions = self.classifier.predict(X)
            return predictions.astype(float)

def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    In practice, you'd load this from a file or database
    """
    spam_messages = [
        "CONGRATULATIONS! You've won $1000! Click here to claim your prize now!",
        "Urgent: Your account will be closed. Verify your information at suspicious-site.com",
        "FREE MONEY! No risk investment opportunity. Call 1-800-SCAM-NOW",
        "Get rich quick! Amazing opportunity! Limited time offer!",
        "Your credit is approved! Get instant cash loan today!",
        "Lose weight fast! Amazing pills! No diet needed! Order now!",
        "Click here for free stuff! Amazing deals! Don't miss out!",
        "Make $5000 per week working from home! No experience needed!",
        "WINNER! You've been selected for a special prize! Claim now!",
        "Debt consolidation! Lower payments! Call now! Limited time!",
        "Free trial! Amazing product! Order today! Money back guarantee!",
        "Hot singles in your area! Meet tonight! Click here now!",
        "Your computer is infected! Download antivirus now! Urgent action required!",
        "Pharmacy online! Cheap medications! No prescription needed! Order now!",
        "Investment opportunity! High returns! No risk! Act fast!"
    ]
    
    ham_messages = [
        "Hi John, can we schedule a meeting for tomorrow at 2 PM?",
        "Thanks for your email. I'll review the document and get back to you.",
        "The quarterly report has been uploaded to the shared folder.",
        "Happy birthday! Hope you have a wonderful day!",
        "Reminder: Team lunch is scheduled for Friday at noon.",
        "Please find the attached invoice for your recent purchase.",
        "The software update has been completed successfully.",
        "Your order has been shipped and will arrive within 3-5 business days.",
        "Thank you for your feedback. We appreciate your business.",
        "The meeting has been rescheduled to Monday at 10 AM.",
        "Your subscription renewal is due next month.",
        "Please review the attached contract and let me know if you have questions.",
        "The system maintenance is scheduled for this weekend.",
        "Your password has been successfully updated.",
        "Thank you for attending yesterday's presentation."
    ]
    
    # Create balanced dataset
    texts = spam_messages + ham_messages
    labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
    
    return texts, labels

def evaluate_model(classifier, X_test, y_test):
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Convert labels for sklearn metrics
    y_test_binary = [1 if label == 'spam' else 0 for label in y_test]
    y_pred_binary = [1 if label == 'spam' else 0 for label in y_pred]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_binary, y_pred_binary, 
                              target_names=['Ham', 'Spam']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return accuracy

def main():
    """
    Main function to demonstrate the spam classifier
    """
    print("COMPLEX SPAM CLASSIFIER FROM SCRATCH")
    print("=" * 50)
    
    # Create sample dataset
    print("Creating sample dataset...")
    texts, labels = create_sample_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Test both algorithms
    algorithms = ['naive_bayes', 'logistic_regression']
    results = {}
    
    for algorithm in algorithms:
        print(f"\n{'='*20} {algorithm.upper()} {'='*20}")
        
        # Train classifier
        classifier = SpamClassifier(algorithm=algorithm)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        print(f"\nEvaluating {algorithm}...")
        accuracy = evaluate_model(classifier, X_test, y_test)
        results[algorithm] = accuracy
        
        # Test with some example messages
        test_messages = [
            "Congratulations! You've won a free iPhone! Click here now!",
            "Can you please send me the meeting notes from yesterday?",
            "URGENT: Your account will be suspended unless you verify now!",
            "Thanks for your help with the project. Much appreciated!"
        ]
        
        print(f"\nTesting {algorithm} with example messages:")
        predictions = classifier.predict(test_messages)
        probabilities = classifier.predict_proba(test_messages)
        
        for i, (message, pred, prob) in enumerate(zip(test_messages, predictions, probabilities)):
            print(f"\nMessage {i+1}: {message[:50]}...")
            print(f"Prediction: {pred} (confidence: {prob:.3f})")
    
    # Compare results
    print(f"\n{'='*20} FINAL RESULTS {'='*20}")
    for algorithm, accuracy in results.items():
        print(f"{algorithm}: {accuracy:.4f}")
    
    print("\nThis implementation demonstrates:")
    print("1. Text preprocessing with spam-specific cleaning")
    print("2. Multiple feature extraction techniques (TF-IDF + hand-crafted)")
    print("3. Two different classification algorithms")
    print("4. Comprehensive evaluation with visualizations")
    print("5. Real-world applicable code structure")

if __name__ == "__main__":
    main()
