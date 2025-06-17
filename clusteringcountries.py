import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style - because ugly plots make data scientists cry
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HealthDataClusterer:
    """
    A comprehensive clustering analysis tool for country health data.
    
    This class handles everything from data preprocessing to visualization,
    because nobody wants to write the same clustering code 50 times.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize our clustering machine.
        
        Args:
            random_state: For reproducible results (because randomness in science is scary)
        """
        self.random_state = random_state
        self.scaler = None
        self.pca = None
        self.clusters = {}
        self.scaled_data = None
        self.original_data = None
        
    def load_sample_data(self):
        """
        Create sample health data for demonstration.
        
        In real life, you'd load this from WHO, World Bank, or similar sources.
        But for now, we're creating realistic fake data because we're fancy like that.
        """
        
        # Country names - a mix of developed, developing, and everything in between
        countries = [
            'United States', 'Germany', 'Japan', 'South Korea', 'United Kingdom',
            'France', 'Canada', 'Australia', 'Switzerland', 'Sweden',
            'Brazil', 'India', 'China', 'Russia', 'Mexico',
            'South Africa', 'Nigeria', 'Kenya', 'Bangladesh', 'Pakistan',
            'Indonesia', 'Thailand', 'Vietnam', 'Philippines', 'Egypt',
            'Turkey', 'Iran', 'Saudi Arabia', 'Israel', 'Argentina'
        ]
        
        np.random.seed(self.random_state)
        n_countries = len(countries)
        
        # Generate realistic health indicators
        # Life expectancy: developed countries higher, with some variation
        life_expectancy = np.random.normal(75, 8, n_countries)
        life_expectancy[:10] += 5  # Boost developed countries
        life_expectancy[15:20] -= 10  # Lower for some developing countries
        
        # Infant mortality: inverse relationship with development
        infant_mortality = 100 - life_expectancy + np.random.normal(0, 5, n_countries)
        infant_mortality = np.maximum(infant_mortality, 2)  # Minimum realistic value
        
        # Healthcare spending per capita (log scale because economics is weird)
        healthcare_spending = np.random.lognormal(7, 1.5, n_countries)
        healthcare_spending[:10] *= 2  # Developed countries spend more
        
        # Doctors per 1000 people
        doctors_per_1000 = np.random.normal(2.5, 1.2, n_countries)
        doctors_per_1000[:10] += 1  # More doctors in developed countries
        doctors_per_1000 = np.maximum(doctors_per_1000, 0.1)
        
        # Hospital beds per 1000 people
        hospital_beds = np.random.normal(4, 2, n_countries)
        hospital_beds = np.maximum(hospital_beds, 0.5)
        
        # Vaccination coverage (percentage)
        vaccination_coverage = np.random.normal(85, 15, n_countries)
        vaccination_coverage[:10] += 5  # Better coverage in developed countries
        vaccination_coverage = np.clip(vaccination_coverage, 30, 98)
        
        # Obesity rate (percentage) - interesting patterns here
        obesity_rate = np.random.normal(20, 8, n_countries)
        obesity_rate[:5] += 10  # Some developed countries have higher obesity
        obesity_rate = np.maximum(obesity_rate, 2)
        
        # Smoking rate (percentage) - varies by culture and policy
        smoking_rate = np.random.normal(22, 8, n_countries)
        smoking_rate = np.clip(smoking_rate, 5, 45)
        
        # Create the dataframe - our data playground
        self.original_data = pd.DataFrame({
            'Country': countries,
            'Life_Expectancy': life_expectancy,
            'Infant_Mortality_Rate': infant_mortality,
            'Healthcare_Spending_Per_Capita': healthcare_spending,
            'Doctors_Per_1000': doctors_per_1000,
            'Hospital_Beds_Per_1000': hospital_beds,
            'Vaccination_Coverage': vaccination_coverage,
            'Obesity_Rate': obesity_rate,
            'Smoking_Rate': smoking_rate
        })
        
        print("📊 Sample health data generated successfully!")
        print(f"Shape: {self.original_data.shape}")
        print("\nFirst few rows:")
        print(self.original_data.head())
        
        return self.original_data
    
    def explore_data(self):
        """
        Explore our data like a curious data scientist.
        
        This is the 'getting to know your data' phase - crucial but often skipped
        by people who think they're too cool for EDA (they're not).
        """
        
        if self.original_data is None:
            print("❌ No data loaded! Call load_sample_data() first.")
            return
        
        print("🔍 DATA EXPLORATION REPORT")
        print("=" * 50)
        
        # Basic statistics - the bread and butter
        print("\n📈 DESCRIPTIVE STATISTICS:")
        numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns
        print(self.original_data[numeric_cols].describe().round(2))
        
        # Check for missing values - the silent data killers
        print("\n🕳️  MISSING VALUES CHECK:")
        missing_counts = self.original_data.isnull().sum()
        if missing_counts.sum() == 0:
            print("✅ No missing values found! (Lucky you)")
        else:
            print(missing_counts[missing_counts > 0])
        
        # Correlation analysis - who's friends with whom?
        print("\n🔗 CORRELATION INSIGHTS:")
        corr_matrix = self.original_data[numeric_cols].corr()
        
        # Find strongest correlations (excluding self-correlations)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
        
        # Sort by absolute correlation strength
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("Strongest correlations:")
        for var1, var2, corr in corr_pairs[:5]:
            direction = "positively" if corr > 0 else "negatively"
            print(f"  • {var1} and {var2}: {corr:.3f} ({direction} correlated)")
        
        # Create correlation heatmap - because humans love colorful squares
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Health Indicators Correlation Matrix\n(Lower triangle only, because symmetry is redundant)')
        plt.tight_layout()
        plt.show()
        
    def preprocess_data(self, scaling_method='standard'):
        """
        Prepare data for clustering - the unsexy but essential step.
        
        Args:
            scaling_method: 'standard' or 'robust' 
                          (robust is better when you have outliers acting up)
        """
        
        if self.original_data is None:
            print("❌ No data to preprocess! Load data first.")
            return
        
        print(f"🔧 PREPROCESSING DATA with {scaling_method} scaling...")
        
        # Extract numeric features (everything except country names)
        numeric_features = self.original_data.select_dtypes(include=[np.number])
        
        # Choose scaler based on preference
        if scaling_method == 'robust':
            # Robust scaler is less sensitive to outliers
            # Uses median and IQR instead of mean and std
            self.scaler = RobustScaler()
            print("Using RobustScaler - good choice for data with outliers!")
        else:
            # Standard scaler - the classic choice
            # Makes features have mean=0 and std=1
            self.scaler = StandardScaler()
            print("Using StandardScaler - the tried and true method!")
        
        # Fit and transform the data
        self.scaled_data = self.scaler.fit_transform(numeric_features)
        
        # Convert back to dataframe for easier handling
        self.scaled_features = pd.DataFrame(
            self.scaled_data,
            columns=numeric_features.columns,
            index=self.original_data.index
        )
        
        print(f"✅ Data scaled successfully! Shape: {self.scaled_data.shape}")
        
        # Show scaling effects
        print("\nScaling effects (mean ± std):")
        for col in numeric_features.columns:
            original_mean = numeric_features[col].mean()
            original_std = numeric_features[col].std()
            scaled_mean = self.scaled_features[col].mean()
            scaled_std = self.scaled_features[col].std()
            print(f"  {col}:")
            print(f"    Original: {original_mean:.2f} ± {original_std:.2f}")
            print(f"    Scaled:   {scaled_mean:.2f} ± {scaled_std:.2f}")
        
    def find_optimal_clusters(self, max_clusters=10):
        """
        Find the optimal number of clusters using multiple methods.
        
        This is like trying to find the perfect number of slices to cut a pizza -
        too few and some people are hungry, too many and it's just messy.
        
        Args:
            max_clusters: Maximum number of clusters to test
        """
        
        if self.scaled_data is None:
            print("❌ Data not preprocessed! Call preprocess_data() first.")
            return
        
        print(f"🎯 FINDING OPTIMAL NUMBER OF CLUSTERS (testing 2-{max_clusters})...")
        
        # Initialize lists to store results
        cluster_range = range(2, max_clusters + 1)
        inertias = []  # Within-cluster sum of squares (lower is better)
        silhouette_scores = []  # How well separated clusters are (higher is better)
        calinski_scores = []  # Another cluster quality metric (higher is better)
        
        # Test different numbers of clusters
        for n_clusters in cluster_range:
            # Fit K-means with current number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_data)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, cluster_labels))
            calinski_scores.append(calinski_harabasz_score(self.scaled_data, cluster_labels))
        
        # Create a comprehensive plot showing all metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cluster Optimization Metrics\n(Multiple perspectives on the same question)', fontsize=16)
        
        # Elbow method plot
        axes[0, 0].plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Elbow Method\n(Look for the "elbow" - where improvement slows)')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia (Within-cluster sum of squares)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Silhouette score plot
        axes[0, 1].plot(cluster_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Silhouette Score\n(Higher is better - well separated clusters)')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz score plot
        axes[1, 0].plot(cluster_range, calinski_scores, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Calinski-Harabasz Score\n(Higher is better - compact and well separated)')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary table
        axes[1, 1].axis('off')
        summary_data = pd.DataFrame({
            'Clusters': cluster_range,
            'Silhouette': [f"{score:.3f}" for score in silhouette_scores],
            'Calinski-H': [f"{score:.1f}" for score in calinski_scores]
        })
        
        # Create table
        table = axes[1, 1].table(cellText=summary_data.values,
                                colLabels=summary_data.columns,
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Summary Table\n(Quick reference for all metrics)')
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal number based on silhouette score
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
        
        print(f"\n🏆 RECOMMENDATION:")
        print(f"Optimal number of clusters: {optimal_clusters}")
        print(f"Best silhouette score: {best_silhouette:.3f}")
        print(f"\nInterpretation:")
        if best_silhouette > 0.7:
            print("• Excellent cluster separation - your data has clear groups!")
        elif best_silhouette > 0.5:
            print("• Good cluster separation - reasonable structure found")
        elif best_silhouette > 0.25:
            print("• Weak cluster separation - clusters exist but overlap")
        else:
            print("• Poor cluster separation - might want to reconsider clustering")
        
        return optimal_clusters
    
    def perform_clustering(self, n_clusters=None, methods=['kmeans', 'hierarchical', 'dbscan']):
        """
        Perform clustering using multiple algorithms.
        
        Like trying different cooking methods for the same ingredients -
        each algorithm has its strengths and quirks.
        
        Args:
            n_clusters: Number of clusters (if None, uses optimal from previous analysis)
            methods: List of clustering methods to try
        """
        
        if self.scaled_data is None:
            print("❌ Data not preprocessed! Call preprocess_data() first.")
            return
        
        if n_clusters is None:
            n_clusters = 3  # Default fallback
            print(f"⚠️  No cluster number specified, using default: {n_clusters}")
        
        print(f"🔬 PERFORMING CLUSTERING with {n_clusters} clusters...")
        print(f"Methods to try: {', '.join(methods)}")
        
        # K-Means Clustering - the classic approach
        if 'kmeans' in methods:
            print("\n1️⃣ K-Means Clustering (the reliable workhorse)...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            kmeans_labels = kmeans.fit_predict(self.scaled_data)
            
            self.clusters['K-Means'] = {
                'labels': kmeans_labels,
                'model': kmeans,
                'silhouette': silhouette_score(self.scaled_data, kmeans_labels),
                'description': 'Partitions data into spherical clusters of similar size'
            }
            
            print(f"   ✅ Silhouette score: {self.clusters['K-Means']['silhouette']:.3f}")
        
        # Hierarchical Clustering - builds a tree of clusters
        if 'hierarchical' in methods:
            print("\n2️⃣ Hierarchical Clustering (builds a family tree of clusters)...")
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            hierarchical_labels = hierarchical.fit_predict(self.scaled_data)
            
            self.clusters['Hierarchical'] = {
                'labels': hierarchical_labels,
                'model': hierarchical,
                'silhouette': silhouette_score(self.scaled_data, hierarchical_labels),
                'description': 'Builds clusters by merging closest points/clusters iteratively'
            }
            
            print(f"   ✅ Silhouette score: {self.clusters['Hierarchical']['silhouette']:.3f}")
        
        # DBSCAN - finds clusters of varying shapes and sizes
        if 'dbscan' in methods:
            print("\n3️⃣ DBSCAN (the rebel - finds weird shaped clusters)...")
            
            # DBSCAN requires parameter tuning - eps (neighborhood size) and min_samples
            # We'll try a few different eps values to find a good one
            eps_values = [0.5, 0.8, 1.0, 1.5, 2.0]
            best_eps = 0.5
            best_score = -1
            best_labels = None
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=3)
                labels = dbscan.fit_predict(self.scaled_data)
                
                # DBSCAN can find noise points (labeled as -1)
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters_found > 1:  # Need at least 2 clusters for silhouette score
                    score = silhouette_score(self.scaled_data, labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_labels = labels
            
            if best_labels is not None:
                n_clusters_found = len(set(best_labels)) - (1 if -1 in best_labels else 0)
                n_noise = list(best_labels).count(-1)
                
                self.clusters['DBSCAN'] = {
                    'labels': best_labels,
                    'model': DBSCAN(eps=best_eps, min_samples=3),
                    'silhouette': best_score,
                    'description': f'Found {n_clusters_found} clusters and {n_noise} noise points'
                }
                
                print(f"   ✅ Best eps: {best_eps}, Silhouette score: {best_score:.3f}")
                print(f"   📊 Found {n_clusters_found} clusters with {n_noise} noise points")
            else:
                print("   ❌ DBSCAN couldn't find meaningful clusters with tested parameters")
        
        # Compare all methods
        print(f"\n🏆 CLUSTERING RESULTS COMPARISON:")
        print("-" * 60)
        for method, results in self.clusters.items():
            print(f"{method:>12}: Silhouette = {results['silhouette']:.3f}")
            print(f"{'':>12}  {results['description']}")
        
        # Find best method
        if self.clusters:
            best_method = max(self.clusters.keys(), key=lambda x: self.clusters[x]['silhouette'])
            print(f"\n🥇 Best performing method: {best_method}")
            
    def analyze_clusters(self, method='K-Means'):
        """
        Analyze and interpret cluster characteristics.
        
        This is where we become cluster detectives, figuring out what makes
        each cluster unique and special (like snowflakes, but with data).
        
        Args:
            method: Which clustering method to analyze
        """
        
        if method not in self.clusters:
            print(f"❌ Method '{method}' not found. Available methods: {list(self.clusters.keys())}")
            return
        
        print(f"🔍 ANALYZING {method} CLUSTERS")
        print("=" * 50)
        
        labels = self.clusters[method]['labels']
        unique_labels = sorted(set(labels))
        
        # Remove noise points for DBSCAN
        if -1 in unique_labels:
            unique_labels.remove(-1)
            noise_points = sum(1 for label in labels if label == -1)
            print(f"Note: {noise_points} points classified as noise (will be excluded from analysis)")
        
        # Create analysis dataframe
        analysis_df = self.original_data.copy()
        analysis_df['Cluster'] = labels
        
        print(f"\n📊 CLUSTER SIZES:")
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        for cluster_id in unique_labels:
            size = cluster_sizes.get(cluster_id, 0)
            percentage = (size / len(labels)) * 100
            print(f"Cluster {cluster_id}: {size} countries ({percentage:.1f}%)")
        
        # Analyze cluster characteristics
        print(f"\n🎯 CLUSTER CHARACTERISTICS:")
        numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns
        
        cluster_means = analysis_df.groupby('Cluster')[numeric_cols].mean()
        
        for cluster_id in unique_labels:
            print(f"\n🏷️  CLUSTER {cluster_id} PROFILE:")
            
            # Get countries in this cluster
            cluster_countries = analysis_df[analysis_df['Cluster'] == cluster_id]['Country'].tolist()
            print(f"Countries ({len(cluster_countries)}): {', '.join(cluster_countries)}")
            
            # Find defining characteristics (features that are notably high or low)
            cluster_mean = cluster_means.loc[cluster_id]
            overall_mean = self.original_data[numeric_cols].mean()
            
            # Calculate z-scores to find distinctive features
            differences = (cluster_mean - overall_mean) / self.original_data[numeric_cols].std()
            
            # Find most distinctive features (absolute z-score > 0.5)
            distinctive_features = differences[abs(differences) > 0.5].sort_values(key=abs, ascending=False)
            
            if len(distinctive_features) > 0:
                print("Key characteristics:")
                for feature, z_score in distinctive_features.items():
                    direction = "above" if z_score > 0 else "below"
                    intensity = "significantly" if abs(z_score) > 1 else "moderately"
                    actual_value = cluster_mean[feature]
                    print(f"  • {feature}: {actual_value:.2f} ({intensity} {direction} average)")
            else:
                print("  • This cluster is close to the overall average across all metrics")
        
        # Create cluster comparison heatmap
        self._create_cluster_heatmap(cluster_means, method)
        
        # Show cluster distribution on first two principal components
        self._visualize_clusters_2d(labels, method)
        
    def _create_cluster_heatmap(self, cluster_means, method):
        """Create a heatmap showing cluster characteristics."""
        
        plt.figure(figsize=(14, 8))
        
        # Normalize data for better visualization (z-score normalization)
        overall_mean = self.original_data.select_dtypes(include=[np.number]).mean()
        overall_std = self.original_data.select_dtypes(include=[np.number]).std()
        normalized_means = (cluster_means - overall_mean) / overall_std
        
        # Create heatmap
        sns.heatmap(normalized_means.T, annot=True, cmap='RdBu_r', center=0,
                    fmt='.2f', cbar_kws={'label': 'Standard deviations from overall mean'})
        
        plt.title(f'{method} Clustering - Cluster Characteristics Heatmap\n' +
                 'Red = Above Average, Blue = Below Average', fontsize=14)
        plt.xlabel('Cluster ID')
        plt.ylabel('Health Indicators')
        plt.tight_layout()
        plt.show()
        
    def _visualize_clusters_2d(self, labels, method):
        """Visualize clusters in 2D using PCA."""
        
        # Apply PCA to reduce dimensionality to 2D for visualization
        if self.pca is None:
            self.pca = PCA(n_components=2, random_state=self.random_state)
            pca_data = self.pca.fit_transform(self.scaled_data)
        else:
            pca_data = self.pca.transform(self.scaled_data)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        unique_labels = sorted(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, cluster_id in enumerate(unique_labels):
            if cluster_id == -1:  # Noise points for DBSCAN
                plt.scatter(pca_data[labels == cluster_id, 0], 
                           pca_data[labels == cluster_id, 1],
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                plt.scatter(pca_data[labels == cluster_id, 0], 
                           pca_data[labels == cluster_id, 1],
                           c=[colors[i]], s=100, alpha=0.7, 
                           label=f'Cluster {cluster_id}')
        
        # Add country labels
        for i, country in enumerate(self.original_data['Country']):
            plt.annotate(country, (pca_data[i, 0], pca_data[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.xlabel(f'First Principal Component (explains {self.pca.explained_variance_ratio_[0]:.1%} of variance)')
        plt.ylabel(f'Second Principal Component (explains {self.pca.explained_variance_ratio_[1]:.1%} of variance)')
        plt.title(f'{method} Clustering Results - 2D Visualization\n' +
                 f'Total variance explained: {sum(self.pca.explained_variance_ratio_):.1%}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def generate_insights(self):
        """
        Generate actionable insights from the clustering analysis.
        
        This is where we put on our consulting hat and pretend we know
        what policy makers should do (spoiler: it's complicated).
        """
        
        if not self.clusters:
            print("❌ No clustering results available. Run clustering first!")
            return
        
        print("💡 STRATEGIC INSIGHTS FROM CLUSTERING ANALYSIS")
        print("=" * 60)
        
        # Find the best performing clustering method
        best_method = max(self.clusters.keys(), key=lambda x: self.clusters[x]['silhouette'])
        labels = self.clusters[best_method]['labels']
        
        print(f"Based on {best_method} clustering (best performing method):\n")
        
        # Analyze each cluster and provide insights
        analysis_df = self.original_data.copy()
        analysis_df['Cluster'] = labels
        
        unique_labels = sorted(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        cluster_insights = {}
        
        for cluster_id in unique_labels:
            cluster_data = analysis_df[analysis_df['Cluster'] == cluster_id]
            cluster_countries = cluster_data['Country'].tolist()
            
            # Calculate cluster characteristics
            cluster_stats = cluster_data.select_dtypes(include=[np.number]).mean()
            
            # Determine cluster archetype based on key metrics
            life_exp = cluster_stats['Life_Expectancy']
            healthcare_spend = cluster_stats['Healthcare_Spending_Per_Capita']
            infant_mort = cluster_stats['Infant_Mortality_Rate']
            
            if life_exp > 78 and healthcare_spend > 3000:
                archetype = "Advanced Healthcare Systems"
                characteristics = "High life expectancy, substantial healthcare investment"
                recommendations = [
                    "Focus on preventive care and healthy aging",
                    "Address lifestyle-related diseases (obesity, smoking)",
                    "Optimize healthcare efficiency and cost-effectiveness"
                ]
            elif life_exp > 70 and healthcare_spend > 1000:
                archetype = "Emerging Healthcare Markets"
                characteristics = "Moderate life expectancy, growing healthcare investment"
                recommendations = [
                    "Strengthen healthcare infrastructure",
                    "Improve vaccination coverage and maternal health",
                    "Invest in medical education and training"
                ]
            elif life_exp < 65 or infant_mort > 50:
                archetype = "Healthcare Development Priority"
                characteristics = "Lower life expectancy, urgent healthcare needs"
                recommendations = [
                    "Focus on basic healthcare access and infrastructure",
                    "Prioritize maternal and child health programs",
                    "Address infectious diseases and malnutrition"
                ]
            else:
                archetype = "Mixed Healthcare Profile"
                characteristics = "Varied healthcare indicators"
                recommendations = [
                    "Conduct detailed individual country assessments",
                    "Implement targeted interventions based on specific needs",
                    "Foster regional healthcare cooperation"
                ]
            
            cluster_insights[cluster_id] = {
                'archetype': archetype,
                'countries': cluster_countries,
                'characteristics': characteristics,
                'recommendations': recommendations,
                'key_stats': {
                    'avg_life_expectancy': life_exp,
                    'avg_healthcare_spending': healthcare_spend,
                    'avg_infant_mortality': infant_mort
                }
            }
        
        # Present insights in a structured way
        for cluster_id, insights in cluster_insights.items():
            print(f"🏥 CLUSTER {cluster_id}: {insights['archetype']}")
            print(f"Countries: {', '.join(insights['countries'])}")
            print(f"Profile: {insights['characteristics']}")
            print(f"Key metrics:")
            print(f"  • Life expectancy: {insights['key_stats']['avg_life_expectancy']:.1f} years")
            print(f"  • Healthcare spending: ${insights['key_stats']['avg_healthcare_spending']:.0f} per capita")
            print(f"  • Infant mortality: {insights['key_stats']['avg_infant_mortality']:.1f} per 1000 births")
            print("Strategic recommendations:")
            for i, rec in enumerate(insights['recommendations'], 1):
                print(f"  {i}. {rec}")
            print()
        
        # Global insights
        print("🌍 GLOBAL HEALTHCARE INSIGHTS:")
        print("-" * 40)
        
        # Calculate global patterns
        total_countries = len([c for c in labels if c != -1])
        high_performers = len([c for c in cluster_insights.values() 
                              if 'Advanced' in c['archetype']])
        
        print(f"• Healthcare development distribution:")
        for cluster_id, insights in cluster_insights.items():
            pct = (len(insights['countries']) / total_countries) * 100
            print(f"  - {insights['archetype']}: {len(insights['countries'])} countries ({pct:.1f}%)")
        
        print(f"\n• Key global challenges identified:")
        if any('Development Priority' in c['archetype'] for c in cluster_insights.values()):
            print("  - Significant healthcare inequality exists globally")
            print("  - Basic healthcare access remains a priority for some regions")
        
        if any('Advanced' in c['archetype'] for c in cluster_insights.values()):
            print("  - Advanced systems need to focus on efficiency and prevention")
            print("  - Opportunity for knowledge transfer to emerging markets")
        
        print(f"\n• Collaboration opportunities:")
        print("  - Regional healthcare partnerships within similar clusters")
        print("  - Technology and knowledge transfer between cluster types")
        print("  - Coordinated global health initiatives targeting specific challenges")
        
    def export_results(self, filename='health_clustering_results'):
        """
        Export clustering results to files for further analysis.
        
        Because what good is analysis if you can't show it to your boss
        or use it in that presentation next week?
        
        Args:
            filename: Base filename for exports (without extension)
        """
        
        if not self.clusters or self.original_data is None:
            print("❌ No results to export! Run clustering analysis first.")
            return
        
        print(f"💾 EXPORTING RESULTS...")
        
        # Find best clustering method
        best_method = max(self.clusters.keys(), key=lambda x: self.clusters[x]['silhouette'])
        labels = self.clusters[best_method]['labels']
        
        # Create comprehensive results dataframe
        results_df = self.original_data.copy()
        results_df['Cluster'] = labels
        results_df['Clustering_Method'] = best_method
        results_df['Silhouette_Score'] = self.clusters[best_method]['silhouette']
        
        # Export main results
        csv_filename = f"{filename}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"✅ Main results exported to: {csv_filename}")
        
        # Create cluster summary
        summary_data = []
        unique_labels = sorted(set(labels))
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns
        
        for cluster_id in unique_labels:
            cluster_data = results_df[results_df['Cluster'] == cluster_id]
            cluster_countries = cluster_data['Country'].tolist()
            
            summary_row = {
                'Cluster_ID': cluster_id,
                'Number_of_Countries': len(cluster_countries),
                'Countries': '; '.join(cluster_countries)
            }
            
            # Add mean values for each numeric column
            for col in numeric_cols:
                summary_row[f'Mean_{col}'] = cluster_data[col].mean()
            
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{filename}_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"✅ Cluster summary exported to: {summary_filename}")
        
        # Export methodology report
        report_filename = f"{filename}_report.txt"
        with open(report_filename, 'w') as f:
            f.write("HEALTH DATA CLUSTERING ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Shape: {self.original_data.shape}\n")
            f.write(f"Best Clustering Method: {best_method}\n")
            f.write(f"Best Silhouette Score: {self.clusters[best_method]['silhouette']:.3f}\n\n")
            
            f.write("CLUSTERING METHODS TESTED:\n")
            for method, results in self.clusters.items():
                f.write(f"- {method}: Silhouette Score = {results['silhouette']:.3f}\n")
                f.write(f"  Description: {results['description']}\n")
            
            f.write(f"\nCLUSTER DISTRIBUTION:\n")
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            for cluster_id in unique_labels:
                size = cluster_sizes.get(cluster_id, 0)
                percentage = (size / len(labels)) * 100
                f.write(f"Cluster {cluster_id}: {size} countries ({percentage:.1f}%)\n")
            
            f.write(f"\nFEATURES ANALYZED:\n")
            for col in numeric_cols:
                f.write(f"- {col}\n")
        
        print(f"✅ Analysis report exported to: {report_filename}")
        print(f"\n📁 All files saved with prefix: {filename}")
        print("Ready for presentation, further analysis, or your data science portfolio!")

def main():
    """
    Main function to demonstrate the complete clustering workflow.
    
    This is your step-by-step guide to clustering countries by health data.
    Think of it as a recipe - follow the steps and you'll get something tasty!
    """
    
    print("🏥 COMPREHENSIVE HEALTH DATA CLUSTERING ANALYSIS")
    print("=" * 60)
    print("Welcome to the world of health data clustering!")
    print("We're about to discover hidden patterns in global health indicators.\n")
    
    # Step 1: Initialize our clustering engine
    print("🚀 Step 1: Initializing clustering analysis engine...")
    clusterer = HealthDataClusterer(random_state=42)
    
    # Step 2: Load and explore the data
    print("\n📊 Step 2: Loading sample health data...")
    data = clusterer.load_sample_data()
    
    print("\n🔍 Step 3: Exploring the data (understanding what we're working with)...")
    clusterer.explore_data()
    
    # Step 3: Preprocess the data
    print("\n🔧 Step 4: Preprocessing data for clustering...")
    clusterer.preprocess_data(scaling_method='standard')
    
    # Step 4: Find optimal number of clusters
    print("\n🎯 Step 5: Finding optimal number of clusters...")
    optimal_k = clusterer.find_optimal_clusters(max_clusters=8)
    
    # Step 5: Perform clustering with multiple methods
    print(f"\n🔬 Step 6: Performing clustering analysis...")
    clusterer.perform_clustering(
        n_clusters=optimal_k, 
        methods=['kmeans', 'hierarchical', 'dbscan']
    )
    
    # Step 6: Analyze clusters in detail
    print(f"\n🔍 Step 7: Analyzing cluster characteristics...")
    best_method = max(clusterer.clusters.keys(), 
                     key=lambda x: clusterer.clusters[x]['silhouette'])
    clusterer.analyze_clusters(method=best_method)
    
    # Step 7: Generate strategic insights
    print(f"\n💡 Step 8: Generating strategic insights...")
    clusterer.generate_insights()
    
    # Step 8: Export results
    print(f"\n💾 Step 9: Exporting results for future use...")
    clusterer.export_results('health_clustering_analysis')
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Congratulations! You've successfully clustered countries by health data.")
    print("The results reveal distinct healthcare archetypes and strategic opportunities.")
    print("\nKey takeaways:")
    print("• Different countries cluster into distinct healthcare development levels")
    print("• Each cluster has unique characteristics and needs different strategies")
    print("• Global health policy can be tailored to these cluster-specific needs")
    print("• Data-driven insights enable more effective healthcare investments")
    
    print("\n📈 Next steps you could take:")
    print("1. Apply this analysis to real WHO or World Bank health data")
    print("2. Include additional health indicators (mental health, disease burden, etc.)")
    print("3. Perform time-series clustering to see how countries evolve")
    print("4. Use these clusters to predict health outcomes or policy effectiveness")
    print("5. Create interactive dashboards for policymakers")
    
    return clusterer

# If running this script directly, execute the main analysis
if __name__ == "__main__":
    # Run the complete analysis
    clustering_results = main()
    
    # Optional: Interactive exploration
    print("\n🎮 INTERACTIVE EXPLORATION:")
    print("The 'clustering_results' object is now available for further exploration!")
    print("Try these commands:")
    print("• clustering_results.original_data.head() - View raw data")
    print("• clustering_results.clusters - See all clustering results")
    print("• clustering_results.analyze_clusters('Hierarchical') - Analyze different method")
    print("• clustering_results.export_results('my_analysis') - Export with custom name")
    
    # Bonus: Quick comparison function
    def quick_compare_methods():
        """Quick function to compare all clustering methods side by side."""
        if clustering_results.clusters:
            print("\n🏆 QUICK METHODS COMPARISON:")
            methods_performance = [(method, results['silhouette']) 
                                 for method, results in clustering_results.clusters.items()]
            methods_performance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (method, score) in enumerate(methods_performance, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                print(f"{medal} {method}: {score:.3f}")
    
    # Run the comparison
    quick_compare_methods()
