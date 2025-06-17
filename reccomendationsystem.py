import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class HybridRecommendationSystem:
    """
    A sophisticated recommendation system that combines multiple approaches:
    - Collaborative Filtering (user-item interactions)
    - Content-Based Filtering (item features)
    - Matrix Factorization (dimensionality reduction)
    - Popularity-based recommendations (fallback for cold start)
    
    This is like having a team of experts each specializing in different aspects
    of recommendation, then combining their opinions for the best results.
    """
    
    def __init__(self, n_components=50, alpha=0.5):
        """
        Initialize our recommendation engine with configurable parameters.
        
        Think of n_components as how many "hidden themes" we want to discover
        in user preferences (like "action movie lovers" or "comedy fans").
        Alpha controls the balance between collaborative and content-based filtering.
        """
        self.n_components = n_components  # For matrix factorization - like finding hidden patterns
        self.alpha = alpha  # Mixing weight between different recommendation strategies
        
        # These will store our trained models - think of them as our "learned knowledge"
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.scaler = StandardScaler()
        
        # Data storage - our system's memory
        self.user_item_matrix = None
        self.item_features_matrix = None
        self.item_similarity_matrix = None
        self.user_factors = None  # User preferences in the reduced space
        self.item_factors = None  # Item characteristics in the reduced space
        
        # Metadata for better recommendations
        self.user_profiles = {}  # What we know about each user
        self.item_profiles = {}  # What we know about each item
        self.popularity_scores = {}  # How popular each item is overall
        
    def fit(self, interactions_df, items_df=None):
        """
        Train our recommendation system on historical data.
        
        This is like teaching the system by showing it thousands of examples
        of what users liked and didn't like, plus information about the items.
        
        interactions_df should have columns: ['user_id', 'item_id', 'rating']
        items_df should have columns: ['item_id', 'features'] where features is text
        """
        print("🔥 Starting to train the recommendation system...")
        print("📚 This is like teaching an AI to understand human preferences!")
        
        # Step 1: Build the user-item interaction matrix
        # This is like creating a giant spreadsheet where rows are users,
        # columns are items, and cells contain ratings (or 0 if no interaction)
        print("\n📊 Building user-item interaction matrix...")
        self.user_item_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        print(f"   Matrix size: {self.user_item_matrix.shape[0]} users × {self.user_item_matrix.shape[1]} items")
        
        # Step 2: Calculate popularity scores for cold-start scenarios
        # Sometimes we need to recommend popular items to new users
        print("\n⭐ Calculating item popularity scores...")
        item_ratings = interactions_df.groupby('item_id').agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        # Popularity combines both rating frequency and average rating
        # It's like asking "What do most people like AND rate highly?"
        for item_id in item_ratings.index:
            count = item_ratings.loc[item_id, ('rating', 'count')]
            avg_rating = item_ratings.loc[item_id, ('rating', 'mean')]
            # Weighted popularity score - more ratings = more reliable
            self.popularity_scores[item_id] = (count * avg_rating) / (count + 5)  # +5 for smoothing
        
        # Step 3: Matrix Factorization using SVD
        # This finds hidden patterns like "users who like action movies also like thrillers"
        print("\n🧠 Performing matrix factorization (finding hidden patterns)...")
        user_item_normalized = self.scaler.fit_transform(self.user_item_matrix.values)
        
        # SVD decomposes our big matrix into smaller, meaningful pieces
        # Think of it as finding the "DNA" of user preferences
        self.user_factors = self.svd_model.fit_transform(user_item_normalized)
        self.item_factors = self.svd_model.components_.T
        
        print(f"   Discovered {self.n_components} hidden preference patterns!")
        
        # Step 4: Content-based filtering (if item features are provided)
        if items_df is not None:
            print("\n📝 Processing item content features...")
            # Convert item descriptions/features into numerical vectors
            # This is like teaching the system to understand what makes items similar
            item_features = items_df.set_index('item_id')['features'].fillna('')
            
            # TF-IDF finds important words that distinguish items from each other
            self.item_features_matrix = self.tfidf_vectorizer.fit_transform(item_features.values)
            
            # Calculate similarity between all items based on their features
            # This helps us say "if you like Item A, you might like Item B because they're similar"
            print("   Computing item-to-item similarities...")
            self.item_similarity_matrix = cosine_similarity(self.item_features_matrix)
            
            # Convert to DataFrame for easier lookup
            feature_items = item_features.index.tolist()
            self.item_similarity_df = pd.DataFrame(
                self.item_similarity_matrix,
                index=feature_items,
                columns=feature_items
            )
        
        # Step 5: Build user and item profiles for better understanding
        print("\n👤 Building user preference profiles...")
        self._build_user_profiles(interactions_df)
        
        print("\n✅ Training complete! The system is now ready to make recommendations.")
        print("🎯 It has learned from user behavior, item similarities, and popularity trends!")
        
    def _build_user_profiles(self, interactions_df):
        """
        Create detailed profiles for each user based on their rating patterns.
        This helps us understand user preferences beyond just the numbers.
        """
        for user_id in interactions_df['user_id'].unique():
            user_data = interactions_df[interactions_df['user_id'] == user_id]
            
            # Build a comprehensive user profile
            profile = {
                'total_ratings': len(user_data),
                'avg_rating': user_data['rating'].mean(),
                'rating_std': user_data['rating'].std(),
                'favorite_items': user_data[user_data['rating'] >= 4]['item_id'].tolist(),
                'disliked_items': user_data[user_data['rating'] <= 2]['item_id'].tolist(),
                'rating_distribution': user_data['rating'].value_counts().to_dict()
            }
            
            self.user_profiles[user_id] = profile
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """
        Generate recommendations based on similar users' preferences.
        
        This is like asking: "What do people with similar taste to this user enjoy?"
        We use the matrix factorization results to find these patterns.
        """
        if user_id not in self.user_item_matrix.index:
            print(f"⚠️  User {user_id} not found in training data. Using popularity-based fallback.")
            return self._get_popularity_recommendations(n_recommendations)
        
        # Get the user's position in our reduced-dimension space
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_vector = self.user_factors[user_idx]
        
        # Calculate predicted ratings for all items
        # This is like asking our trained model: "What would this user rate each item?"
        predicted_ratings = np.dot(user_vector, self.item_factors.T)
        
        # Get items the user hasn't interacted with yet
        user_rated_items = set(self.user_item_matrix.columns[self.user_item_matrix.iloc[user_idx] > 0])
        all_items = set(self.user_item_matrix.columns)
        unrated_items = all_items - user_rated_items
        
        # Create recommendations with predicted scores
        recommendations = []
        for item_id in unrated_items:
            if item_id in self.user_item_matrix.columns:
                item_idx = self.user_item_matrix.columns.get_loc(item_id)
                predicted_score = predicted_ratings[item_idx]
                recommendations.append((item_id, predicted_score))
        
        # Sort by predicted rating and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_content_recommendations(self, user_id, n_recommendations=10):
        """
        Generate recommendations based on item content similarity.
        
        This asks: "What items are similar to things this user has liked before?"
        """
        if self.item_similarity_matrix is None:
            print("⚠️  No content features available. Falling back to collaborative filtering.")
            return self.get_collaborative_recommendations(user_id, n_recommendations)
        
        if user_id not in self.user_profiles:
            return self._get_popularity_recommendations(n_recommendations)
        
        # Get user's favorite items (highly rated)
        favorite_items = self.user_profiles[user_id]['favorite_items']
        
        if not favorite_items:
            print(f"⚠️  No clear favorites found for user {user_id}. Using popularity fallback.")
            return self._get_popularity_recommendations(n_recommendations)
        
        # Find items similar to user's favorites
        item_scores = {}
        
        for fav_item in favorite_items:
            if fav_item in self.item_similarity_df.index:
                # Get similarity scores for this favorite item
                similar_items = self.item_similarity_df.loc[fav_item]
                
                # Add to running tally (items similar to multiple favorites get higher scores)
                for item_id, similarity in similar_items.items():
                    if item_id != fav_item and item_id not in favorite_items:
                        item_scores[item_id] = item_scores.get(item_id, 0) + similarity
        
        # Sort and return top recommendations
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_hybrid_recommendations(self, user_id, n_recommendations=10):
        """
        The main recommendation method that combines all our approaches.
        
        This is like having a panel of experts vote on what to recommend,
        then combining their opinions intelligently.
        """
        print(f"\n🎯 Generating hybrid recommendations for user {user_id}...")
        
        # Get recommendations from both collaborative and content-based methods
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)  # Get more to have options
        content_recs = self.get_content_recommendations(user_id, n_recommendations * 2)
        
        # Combine the recommendations with weighted scoring
        combined_scores = {}
        
        # Add collaborative filtering scores
        print("   🤝 Incorporating collaborative filtering insights...")
        for item_id, score in collab_recs:
            combined_scores[item_id] = self.alpha * score
        
        # Add content-based scores
        print("   📋 Adding content-based similarity scores...")
        for item_id, score in content_recs:
            if item_id in combined_scores:
                combined_scores[item_id] += (1 - self.alpha) * score
            else:
                combined_scores[item_id] = (1 - self.alpha) * score
        
        # Boost popular items slightly (helps with quality)
        print("   ⭐ Applying popularity boost...")
        for item_id in combined_scores:
            if item_id in self.popularity_scores:
                # Small boost based on popularity (prevents recommending obscure items)
                popularity_boost = 0.1 * self.popularity_scores[item_id]
                combined_scores[item_id] += popularity_boost
        
        # Sort by final combined score
        final_recommendations = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"   ✅ Generated {len(final_recommendations)} hybrid recommendations!")
        return final_recommendations[:n_recommendations]
    
    def _get_popularity_recommendations(self, n_recommendations=10):
        """
        Fallback method for new users or when other methods fail.
        Simply recommends the most popular items.
        """
        print("   📈 Using popularity-based recommendations...")
        popular_items = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        return popular_items[:n_recommendations]
    
    def explain_recommendation(self, user_id, item_id):
        """
        Provide human-readable explanation for why an item was recommended.
        This helps build trust and understanding with users.
        """
        explanation = f"Why we recommended item '{item_id}' for user '{user_id}':\n"
        
        # Check user profile
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            explanation += f"• You typically rate items {profile['avg_rating']:.1f}/5 on average\n"
            
            # Check if similar to user's favorites
            if self.item_similarity_df is not None and item_id in self.item_similarity_df.index:
                max_similarity = 0
                similar_favorite = None
                
                for fav_item in profile['favorite_items']:
                    if fav_item in self.item_similarity_df.index:
                        similarity = self.item_similarity_df.loc[item_id, fav_item]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            similar_favorite = fav_item
                
                if similar_favorite and max_similarity > 0.3:
                    explanation += f"• This item is {max_similarity:.1%} similar to '{similar_favorite}' which you rated highly\n"
        
        # Add popularity context
        if item_id in self.popularity_scores:
            pop_score = self.popularity_scores[item_id]
            explanation += f"• This item has a popularity score of {pop_score:.2f} (higher = more universally liked)\n"
        
        explanation += "• Our AI model predicts you'll enjoy this based on users with similar preferences"
        
        return explanation
    
    def get_user_insights(self, user_id):
        """
        Provide insights about a user's preferences and behavior.
        This is useful for understanding recommendation patterns.
        """
        if user_id not in self.user_profiles:
            return f"No data available for user {user_id}"
        
        profile = self.user_profiles[user_id]
        
        insights = f"""
🔍 User {user_id} Preference Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Rating Behavior:
   • Total items rated: {profile['total_ratings']}
   • Average rating given: {profile['avg_rating']:.2f}/5
   • Rating consistency: {'High' if profile['rating_std'] < 1.0 else 'Moderate' if profile['rating_std'] < 1.5 else 'Low'}

❤️  Favorites ({len(profile['favorite_items'])} items):
   • Items rated 4+ stars: {', '.join(map(str, profile['favorite_items'][:5]))}{'...' if len(profile['favorite_items']) > 5 else ''}

👎 Dislikes ({len(profile['disliked_items'])} items):
   • Items rated 2 or below: {', '.join(map(str, profile['disliked_items'][:5]))}{'...' if len(profile['disliked_items']) > 5 else ''}

📈 Rating Distribution:
"""
        
        for rating, count in sorted(profile['rating_distribution'].items()):
            percentage = (count / profile['total_ratings']) * 100
            bar = '█' * int(percentage // 5)  # Simple bar chart
            insights += f"   {rating}⭐: {count} ratings ({percentage:.1f}%) {bar}\n"
        
        return insights


# Example usage and demonstration
def demo_recommendation_system():
    """
    Demonstrate how to use our sophisticated recommendation system.
    This creates sample data and shows all the features in action.
    """
    print("🚀 ADVANCED RECOMMENDATION SYSTEM DEMO")
    print("=" * 50)
    
    # Create sample interaction data
    # In reality, this would come from your database or data files
    print("\n📝 Creating sample user-item interaction data...")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic sample data
    users = [f'user_{i}' for i in range(1, 101)]  # 100 users
    items = [f'movie_{i}' for i in range(1, 51)]   # 50 movies
    
    interactions = []
    for user in users:
        # Each user rates between 5-25 movies (realistic range)
        n_ratings = np.random.randint(5, 26)
        rated_items = np.random.choice(items, n_ratings, replace=False)
        
        for item in rated_items:
            # Generate ratings with some user bias (some users are more generous)
            user_bias = np.random.normal(0, 0.5)  # User's tendency to rate higher/lower
            base_rating = 3.5  # Average movie rating
            rating = np.clip(np.random.normal(base_rating + user_bias, 1.0), 1, 5)
            
            interactions.append({
                'user_id': user,
                'item_id': item, 
                'rating': round(rating, 1)
            })
    
    interactions_df = pd.DataFrame(interactions)
    print(f"   Generated {len(interactions)} user-item interactions")
    
    # Create sample item features (movie descriptions)
    print("\n🎬 Creating sample movie content features...")
    
    # Sample movie genres and themes for content-based filtering
    genres = ['action', 'comedy', 'drama', 'thriller', 'romance', 'sci-fi', 'horror', 'adventure']
    themes = ['family', 'friendship', 'love', 'betrayal', 'heroism', 'mystery', 'war', 'crime']
    
    items_data = []
    for item in items:
        # Each movie has 2-4 genres/themes
        movie_features = np.random.choice(genres + themes, np.random.randint(2, 5), replace=False)
        features_text = ' '.join(movie_features)
        
        items_data.append({
            'item_id': item,
            'features': features_text
        })
    
    items_df = pd.DataFrame(items_data)
    print(f"   Created feature descriptions for {len(items_data)} movies")
    
    # Initialize and train the recommendation system
    print("\n🤖 Initializing the Advanced Recommendation System...")
    rec_system = HybridRecommendationSystem(n_components=20, alpha=0.6)
    
    # Train the system
    rec_system.fit(interactions_df, items_df)
    
    # Demonstrate different types of recommendations
    test_user = 'user_1'
    
    print(f"\n🎯 RECOMMENDATION RESULTS FOR {test_user.upper()}")
    print("=" * 50)
    
    # Show user insights first
    print(rec_system.get_user_insights(test_user))
    
    # Get hybrid recommendations
    print(f"\n🏆 TOP HYBRID RECOMMENDATIONS:")
    hybrid_recs = rec_system.get_hybrid_recommendations(test_user, 5)
    
    for i, (item_id, score) in enumerate(hybrid_recs, 1):
        print(f"\n{i}. {item_id} (Score: {score:.3f})")
        print("   " + rec_system.explain_recommendation(test_user, item_id))
    
    # Compare different recommendation approaches
    print(f"\n📊 COMPARING RECOMMENDATION APPROACHES:")
    print("-" * 50)
    
    collab_recs = rec_system.get_collaborative_recommendations(test_user, 3)
    print(f"\n🤝 Collaborative Filtering Top 3:")
    for item_id, score in collab_recs:
        print(f"   • {item_id}: {score:.3f}")
    
    content_recs = rec_system.get_content_recommendations(test_user, 3)
    print(f"\n📋 Content-Based Filtering Top 3:")
    for item_id, score in content_recs:
        print(f"   • {item_id}: {score:.3f}")
    
    print(f"\n🎯 Hybrid (Combined) Top 3:")
    for item_id, score in hybrid_recs[:3]:
        print(f"   • {item_id}: {score:.3f}")
    
    print("\n✅ Demo completed! The system successfully combines multiple AI techniques")
    print("   to provide personalized, explainable recommendations! 🎉")


if __name__ == "__main__":
    # Run the demonstration
    demo_recommendation_system()
