import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ComplexAnomalyDetector:
    """
    Okay, so here's the thing - anomaly detection is like being a detective.
    You're looking for things that just don't fit the normal pattern.
    This class combines multiple approaches because, honestly, no single method
    is perfect for every situation. It's like having multiple tools in your toolbox.
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        # contamination is basically "how much weird stuff do we expect?"
        # 0.1 means we expect about 10% of our data to be anomalies
        # This is a reasonable starting point, but you might need to adjust
        self.contamination = contamination
        self.random_state = random_state
        
        # We'll store our fitted models here - think of it as our trained detectives
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        
        # Keep track of feature names so we can explain what's happening
        self.feature_names = None
        
        print(f"🕵️ Anomaly Detective initialized! Expecting {contamination*100}% suspicious activity...")
    
    def _prepare_data(self, X, fit_scalers=True):
        """
        Data prep is like cleaning your glasses before looking for clues.
        Different algorithms work better when data is scaled properly.
        Some are sensitive to outliers, others aren't - so we use different scalers.
        """
        
        # Convert to DataFrame if it's not already (makes life easier)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Store feature names for later explanation
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        # Handle missing values - can't detect anomalies in missing data!
        if X.isnull().any().any():
            print("⚠️ Found some missing values. Filling with median values...")
            X = X.fillna(X.median())
        
        scaled_data = {}
        
        if fit_scalers:
            # StandardScaler: assumes normal distribution, sensitive to outliers
            # Good for algorithms that assume normality
            self.scalers['standard'] = StandardScaler()
            scaled_data['standard'] = self.scalers['standard'].fit_transform(X)
            
            # RobustScaler: uses median and IQR, less sensitive to outliers
            # Better when we already suspect there might be anomalies
            self.scalers['robust'] = RobustScaler()
            scaled_data['robust'] = self.scalers['robust'].fit_transform(X)
        else:
            # We're in prediction mode, use already fitted scalers
            scaled_data['standard'] = self.scalers['standard'].transform(X)
            scaled_data['robust'] = self.scalers['robust'].transform(X)
        
        return X, scaled_data
    
    def fit(self, X):
        """
        This is where we train our army of anomaly detectors.
        Each one has different strengths - like having specialists on a team.
        """
        
        print("🎯 Training anomaly detection models...")
        
        # Prepare the data with different scaling approaches
        X_original, scaled_data = self._prepare_data(X, fit_scalers=True)
        
        # Method 1: Isolation Forest
        # Think of this as randomly chopping up the data space with cuts
        # Anomalies are easier to isolate (fewer cuts needed)
        # Works well with high-dimensional data and doesn't assume any distribution
        print("🌲 Training Isolation Forest (the tree-chopping detective)...")
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100  # More trees = more stable results
        )
        self.models['isolation_forest'].fit(scaled_data['robust'])
        
        # Method 2: One-Class SVM
        # Creates a boundary around normal data using support vectors
        # Good at finding complex decision boundaries
        # Can be slow on large datasets but very effective
        print("🎯 Training One-Class SVM (the boundary-drawing detective)...")
        self.models['one_class_svm'] = OneClassSVM(
            nu=self.contamination,  # nu is roughly equivalent to contamination
            kernel='rbf',  # Radial basis function - good for non-linear patterns
            gamma='scale'  # Let sklearn figure out the right gamma
        )
        self.models['one_class_svm'].fit(scaled_data['standard'])
        
        # Method 3: Elliptic Envelope (Robust Covariance)
        # Assumes data follows a multivariate Gaussian distribution
        # Finds the "normal" ellipse and flags points outside it
        # Works well when your normal data is roughly bell-shaped
        print("🥚 Training Elliptic Envelope (the shape-fitting detective)...")
        self.models['elliptic_envelope'] = EllipticEnvelope(
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.models['elliptic_envelope'].fit(scaled_data['standard'])
        
        # Method 4: Local Outlier Factor
        # Looks at local density - are you in a sparse neighborhood?
        # Great for finding local anomalies that might be normal globally
        # Only works in fit_predict mode, so we'll handle it differently
        print("🏘️ Preparing Local Outlier Factor (the neighborhood watch)...")
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination
        )
        # LOF doesn't have a separate fit method, so we store the training data
        self.lof_training_data = scaled_data['robust'].copy()
        
        # Method 5: Statistical approach using Z-score
        # Simple but effective for univariate outliers
        # We'll use this for feature-level analysis
        print("📊 Computing statistical baselines (the number-crunching detective)...")
        self.statistical_stats = {}
        for i, col in enumerate(self.feature_names):
            self.statistical_stats[col] = {
                'mean': np.mean(scaled_data['standard'][:, i]),
                'std': np.std(scaled_data['standard'][:, i]),
                'q1': np.percentile(scaled_data['standard'][:, i], 25),
                'q3': np.percentile(scaled_data['standard'][:, i], 75)
            }
        
        # Store original data stats for interpretation
        self.original_stats = X_original.describe()
        
        self.is_fitted = True
        print("✅ All detectives are trained and ready for action!")
        
        return self
    
    def predict(self, X):
        """
        Now we put our trained detectives to work!
        Each one votes on whether each point is an anomaly.
        We'll combine their votes for a final decision.
        """
        
        if not self.is_fitted:
            raise ValueError("🚨 Hold up! You need to fit the model first. Call fit() before predict().")
        
        print("🔍 Running anomaly detection analysis...")
        
        # Prepare the data (but don't refit scalers)
        X_original, scaled_data = self._prepare_data(X, fit_scalers=False)
        
        # Get predictions from each model
        predictions = {}
        scores = {}
        
        # Isolation Forest
        predictions['isolation_forest'] = self.models['isolation_forest'].predict(scaled_data['robust'])
        scores['isolation_forest'] = self.models['isolation_forest'].score_samples(scaled_data['robust'])
        
        # One-Class SVM
        predictions['one_class_svm'] = self.models['one_class_svm'].predict(scaled_data['standard'])
        scores['one_class_svm'] = self.models['one_class_svm'].score_samples(scaled_data['standard'])
        
        # Elliptic Envelope
        predictions['elliptic_envelope'] = self.models['elliptic_envelope'].predict(scaled_data['standard'])
        scores['elliptic_envelope'] = self.models['elliptic_envelope'].score_samples(scaled_data['standard'])
        
        # Local Outlier Factor (needs special handling)
        # We need to combine training and test data for LOF
        combined_data = np.vstack([self.lof_training_data, scaled_data['robust']])
        lof_predictions = self.models['lof'].fit_predict(combined_data)
        predictions['lof'] = lof_predictions[-len(X_original):]  # Get only test predictions
        
        # LOF scores (negative outlier factor)
        lof_scores = self.models['lof'].negative_outlier_factor_
        scores['lof'] = lof_scores[-len(X_original):]
        
        # Statistical approach - flag points that are unusual in multiple dimensions
        statistical_anomalies = np.zeros(len(X_original))
        for i, col in enumerate(self.feature_names):
            col_data = scaled_data['standard'][:, i]
            # Use both Z-score and IQR methods
            z_scores = np.abs((col_data - self.statistical_stats[col]['mean']) / self.statistical_stats[col]['std'])
            
            # IQR method
            q1, q3 = self.statistical_stats[col]['q1'], self.statistical_stats[col]['q3']
            iqr = q3 - q1
            iqr_outliers = (col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))
            
            # Combine both statistical methods
            statistical_anomalies += (z_scores > 3) | iqr_outliers
        
        # Convert to -1/1 format to match other predictors
        predictions['statistical'] = np.where(statistical_anomalies >= 1, -1, 1)
        scores['statistical'] = -statistical_anomalies  # Negative because more violations = more anomalous
        
        # Ensemble voting: combine all predictions
        # Convert predictions to a more workable format (0/1 instead of -1/1)
        pred_matrix = np.array([np.where(pred == -1, 1, 0) for pred in predictions.values()]).T
        
        # Majority vote with some weighting
        # Isolation Forest and One-Class SVM tend to be more reliable
        weights = {
            'isolation_forest': 1.2,
            'one_class_svm': 1.2,
            'elliptic_envelope': 1.0,
            'lof': 1.1,
            'statistical': 0.8
        }
        
        weighted_votes = np.zeros(len(X_original))
        total_weight = 0
        
        for i, (method, pred) in enumerate(predictions.items()):
            anomaly_votes = np.where(pred == -1, 1, 0)
            weighted_votes += anomaly_votes * weights[method]
            total_weight += weights[method]
        
        # Final ensemble prediction
        ensemble_score = weighted_votes / total_weight
        ensemble_prediction = np.where(ensemble_score > 0.5, -1, 1)
        
        # Create a comprehensive results dictionary
        results = {
            'ensemble_prediction': ensemble_prediction,
            'ensemble_score': ensemble_score,
            'individual_predictions': predictions,
            'individual_scores': scores,
            'anomaly_probability': ensemble_score,  # Easier to interpret name
            'is_anomaly': ensemble_prediction == -1
        }
        
        # Add some interpretability
        results['feature_analysis'] = self._analyze_features(X_original, scaled_data['standard'])
        
        print(f"🎯 Analysis complete! Found {np.sum(ensemble_prediction == -1)} potential anomalies out of {len(X_original)} samples")
        
        return results
    
    def _analyze_features(self, X_original, X_scaled):
        """
        This is where we try to explain WHY something might be anomalous.
        It's like a detective explaining their reasoning.
        """
        
        feature_analysis = []
        
        for idx in range(len(X_original)):
            sample_analysis = {}
            
            for i, feature in enumerate(self.feature_names):
                original_value = X_original.iloc[idx, i]
                scaled_value = X_scaled[idx, i]
                
                # Calculate how unusual this value is
                z_score = abs((scaled_value - self.statistical_stats[feature]['mean']) / 
                             self.statistical_stats[feature]['std'])
                
                # Compare to the distribution of training data
                percentile = stats.percentileofscore(
                    X_scaled[:, i], scaled_value
                )
                
                sample_analysis[feature] = {
                    'original_value': original_value,
                    'z_score': z_score,
                    'percentile': percentile,
                    'is_extreme': z_score > 2.5  # More than 2.5 standard deviations
                }
            
            feature_analysis.append(sample_analysis)
        
        return feature_analysis
    
    def explain_anomaly(self, X, sample_idx):
        """
        Explain why a specific sample was flagged as anomalous.
        This is the detective presenting their case!
        """
        
        if not self.is_fitted:
            raise ValueError("Model needs to be fitted first!")
        
        results = self.predict(X)
        
        if sample_idx >= len(X):
            raise ValueError(f"Sample index {sample_idx} is out of range!")
        
        sample_analysis = results['feature_analysis'][sample_idx]
        is_anomaly = results['is_anomaly'][sample_idx]
        anomaly_score = results['anomaly_probability'][sample_idx]
        
        print(f"\n🔍 ANOMALY ANALYSIS FOR SAMPLE {sample_idx}")
        print("=" * 50)
        
        if is_anomaly:
            print(f"🚨 VERDICT: ANOMALY DETECTED (confidence: {anomaly_score:.2%})")
        else:
            print(f"✅ VERDICT: NORMAL SAMPLE (confidence: {1-anomaly_score:.2%})")
        
        print(f"\n📊 INDIVIDUAL DETECTOR VOTES:")
        for method, prediction in results['individual_predictions'].items():
            vote = "🚨 ANOMALY" if prediction[sample_idx] == -1 else "✅ NORMAL"
            score = results['individual_scores'][method][sample_idx]
            print(f"   {method.replace('_', ' ').title()}: {vote} (score: {score:.3f})")
        
        print(f"\n🎯 FEATURE-BY-FEATURE BREAKDOWN:")
        extreme_features = []
        
        for feature, analysis in sample_analysis.items():
            status = "🔥 EXTREME" if analysis['is_extreme'] else "📊 normal"
            print(f"   {feature}: {analysis['original_value']:.3f} "
                  f"(z-score: {analysis['z_score']:.2f}, "
                  f"percentile: {analysis['percentile']:.1f}%) {status}")
            
            if analysis['is_extreme']:
                extreme_features.append(feature)
        
        if extreme_features and is_anomaly:
            print(f"\n💡 LIKELY REASONS FOR ANOMALY:")
            print(f"   The following features have unusual values: {', '.join(extreme_features)}")
            
            # Give some context about what "unusual" means
            for feature in extreme_features:
                analysis = sample_analysis[feature]
                if analysis['percentile'] < 5:
                    print(f"   - {feature} is unusually LOW (bottom 5% of values)")
                elif analysis['percentile'] > 95:
                    print(f"   - {feature} is unusually HIGH (top 5% of values)")
                else:
                    print(f"   - {feature} has an unusual combination with other features")
    
    def plot_anomalies(self, X, results=None, figsize=(15, 10)):
        """
        Create visualizations to help understand the anomalies.
        A picture is worth a thousand words, especially in data science!
        """
        
        if results is None:
            results = self.predict(X)
        
        # Convert X to DataFrame if it isn't already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Create a comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('🕵️ Anomaly Detection Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Anomaly distribution
        ax1 = axes[0, 0]
        anomaly_counts = [np.sum(results['is_anomaly']), np.sum(~results['is_anomaly'])]
        colors = ['#ff6b6b', '#4ecdc4']
        ax1.pie(anomaly_counts, labels=['Anomalies', 'Normal'], colors=colors, autopct='%1.1f%%')
        ax1.set_title('Distribution of Anomalies')
        
        # Plot 2: Anomaly scores distribution
        ax2 = axes[0, 1]
        ax2.hist(results['anomaly_probability'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0.5, color='red', linestyle='--', label='Anomaly Threshold')
        ax2.set_xlabel('Anomaly Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Anomaly Score Distribution')
        ax2.legend()
        
        # Plot 3: Method agreement
        ax3 = axes[0, 2]
        method_names = list(results['individual_predictions'].keys())
        method_counts = [np.sum(np.array(pred) == -1) for pred in results['individual_predictions'].values()]
        
        bars = ax3.bar(range(len(method_names)), method_counts, color='lightcoral')
        ax3.set_xticks(range(len(method_names)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in method_names], rotation=45, ha='right')
        ax3.set_ylabel('Anomalies Detected')
        ax3.set_title('Detection Method Comparison')
        
        # Plot 4: Feature correlation with anomalies (if we have numeric features)
        ax4 = axes[1, 0]
        if len(self.feature_names) >= 2:
            # Use first two features for scatter plot
            feature1, feature2 = self.feature_names[0], self.feature_names[1]
            
            normal_mask = ~results['is_anomaly']
            anomaly_mask = results['is_anomaly']
            
            ax4.scatter(X.loc[normal_mask, feature1], X.loc[normal_mask, feature2], 
                       c='blue', alpha=0.6, label='Normal', s=50)
            ax4.scatter(X.loc[anomaly_mask, feature1], X.loc[anomaly_mask, feature2], 
                       c='red', alpha=0.8, label='Anomaly', s=100, marker='^')
            
            ax4.set_xlabel(feature1)
            ax4.set_ylabel(feature2)
            ax4.set_title(f'{feature1} vs {feature2}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Need at least 2 features\nfor scatter plot', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Scatter Plot')
        
        # Plot 5: Anomaly scores by method
        ax5 = axes[1, 1]
        score_data = []
        method_labels = []
        
        for method, scores in results['individual_scores'].items():
            score_data.append(scores)
            method_labels.append(method.replace('_', ' ').title())
        
        bp = ax5.boxplot(score_data, labels=method_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax5.set_ylabel('Anomaly Scores')
        ax5.set_title('Score Distribution by Method')
        ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Top anomalous samples
        ax6 = axes[1, 2]
        top_anomalies_idx = np.argsort(results['anomaly_probability'])[-10:]
        top_scores = results['anomaly_probability'][top_anomalies_idx]
        
        bars = ax6.barh(range(len(top_anomalies_idx)), top_scores, color='salmon')
        ax6.set_yticks(range(len(top_anomalies_idx)))
        ax6.set_yticklabels([f'Sample {idx}' for idx in top_anomalies_idx])
        ax6.set_xlabel('Anomaly Probability')
        ax6.set_title('Top 10 Most Anomalous Samples')
        
        plt.tight_layout()
        plt.show()
        
        # Print some summary statistics
        print(f"\n📈 DETECTION SUMMARY:")
        print(f"   Total samples analyzed: {len(X)}")
        print(f"   Anomalies detected: {np.sum(results['is_anomaly'])} ({np.mean(results['is_anomaly']):.1%})")
        print(f"   Average anomaly probability: {np.mean(results['anomaly_probability']):.3f}")
        print(f"   Most anomalous sample: #{np.argmax(results['anomaly_probability'])} "
              f"(probability: {np.max(results['anomaly_probability']):.3f})")

# Example usage and demonstration
def demonstrate_anomaly_detection():
    """
    Let's create some sample data and show how this detector works.
    This is like a demo for our detective agency!
    """
    
    print("🎭 ANOMALY DETECTION DEMONSTRATION")
    print("=" * 50)
    
    # Generate some sample data with known anomalies
    np.random.seed(42)
    
    # Normal data: customers with typical behavior
    normal_data = np.random.multivariate_normal(
        mean=[100, 50, 25],  # average purchase amount, frequency, satisfaction
        cov=[[400, 50, 10], [50, 100, 5], [10, 5, 25]],
        size=800
    )
    
    # Anomalous data: unusual customer behavior
    anomalies = np.array([
        [500, 200, 95],  # High-value, high-frequency customer (VIP)
        [10, 5, 5],      # Low-value, low-frequency, low satisfaction
        [200, 10, 90],   # High-value but infrequent purchases
        [50, 100, 10],   # Frequent but low-value, low satisfaction
        [300, 300, 50],  # Extremely high frequency
        [-10, 20, 80],   # Negative purchase amount (returns?)
        [150, 50, 110],  # Satisfaction score out of normal range
        [1000, 5, 95],   # Extremely high single purchase
    ])
    
    # Combine the data
    all_data = np.vstack([normal_data, anomalies])
    
    # Create feature names
    feature_names = ['Purchase_Amount', 'Purchase_Frequency', 'Satisfaction_Score']
    df = pd.DataFrame(all_data, columns=feature_names)
    
    print(f"📊 Generated {len(df)} samples with {len(anomalies)} known anomalies")
    print("Features:", feature_names)
    
    # Initialize and train the detector
    detector = ComplexAnomalyDetector(contamination=0.1)
    detector.fit(df)
    
    # Detect anomalies
    results = detector.predict(df)
    
    # Show results
    print(f"\n🎯 DETECTION RESULTS:")
    detected_anomalies = np.sum(results['is_anomaly'])
    print(f"   Detected {detected_anomalies} anomalies")
    
    # Check how many of our known anomalies were caught
    known_anomaly_indices = list(range(len(normal_data), len(df)))
    caught_known_anomalies = np.sum(results['is_anomaly'][known_anomaly_indices])
    
    print(f"   Caught {caught_known_anomalies} out of {len(anomalies)} known anomalies")
    print(f"   Detection rate: {caught_known_anomalies/len(anomalies):.1%}")
    
    # Explain a few interesting cases
    if detected_anomalies > 0:
        print(f"\n🔍 Let's examine some detected anomalies...")
        anomaly_indices = np.where(results['is_anomaly'])[0]
        
        # Explain the most anomalous sample
        most_anomalous_idx = np.argmax(results['anomaly_probability'])
        detector.explain_anomaly(df, most_anomalous_idx)
        
        # Create visualizations
        detector.plot_anomalies(df, results)
    
    return detector, df, results

# Run the demonstration if this script is executed directly
if __name__ == "__main__":
    # This is where the magic happens!
    detector, sample_data, results = demonstrate_anomaly_detection()
    
    print(f"\n🎉 Demo complete! The detector is ready to use.")
    print("💡 You can now use it on your own data like this:")
    print("   detector = ComplexAnomalyDetector()")
    print("   detector.fit(your_training_data)")
    print("   results = detector.predict(your_test_data)")
    print("   detector.explain_anomaly(your_test_data, sample_index)")
