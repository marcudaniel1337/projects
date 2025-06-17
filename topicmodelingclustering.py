# Topic Modeling and Clustering with BERTopic in Python
# A comprehensive guide with human-like explanations

# First, let's install the necessary packages
# BERTopic is fantastic because it combines the power of transformers with traditional clustering
"""
pip install bertopic
pip install umap-learn
pip install hdbscan
pip install sentence-transformers
pip install plotly
pip install pandas
pip install scikit-learn
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Let's create some sample data to work with
# In real life, you'd load your own dataset, but this gives us something to play with
sample_documents = [
    "Machine learning is revolutionizing the tech industry with advanced algorithms",
    "Artificial intelligence and deep learning are transforming healthcare diagnostics",
    "Neural networks and computer vision enable autonomous vehicle navigation",
    "Natural language processing helps chatbots understand human communication",
    "Climate change is causing unprecedented global warming and weather patterns",
    "Renewable energy sources like solar and wind power are becoming more efficient",
    "Carbon emissions from fossil fuels contribute to environmental degradation",
    "Sustainable development goals focus on environmental conservation",
    "Stock markets are experiencing volatility due to economic uncertainty",
    "Cryptocurrency and blockchain technology are disrupting traditional finance",
    "Interest rates and inflation affect consumer spending and investment decisions",
    "Global trade policies impact international business and economic growth",
    "Social media platforms influence public opinion and political discourse",
    "Digital privacy concerns arise from data collection by tech companies",
    "Cybersecurity threats target personal information and corporate databases",
    "Online education platforms have expanded access to learning resources",
    "COVID-19 pandemic has accelerated remote work and digital transformation",
    "Vaccine development and public health measures help control disease spread",
    "Mental health awareness has increased during challenging times",
    "Healthcare systems are adapting to provide better patient care"
] * 10  # Multiply by 10 to have more documents for better clustering

print(f"Working with {len(sample_documents)} documents")

# =============================================================================
# STEP 1: BASIC BERTOPIC IMPLEMENTATION
# =============================================================================

# The beauty of BERTopic is that it can work with minimal setup
# But let's understand what's happening under the hood
print("\n=== BASIC BERTOPIC IMPLEMENTATION ===")

# Create a basic BERTopic model
# This uses default settings: all-MiniLM-L6-v2 for embeddings, UMAP for dimensionality reduction, 
# and HDBSCAN for clustering
basic_topic_model = BERTopic(verbose=True)

# Fit the model and transform our documents
# This does several things:
# 1. Creates embeddings for each document using sentence transformers
# 2. Reduces dimensionality with UMAP (high-dimensional -> 2D typically)
# 3. Clusters the reduced embeddings with HDBSCAN
# 4. Extracts topics using c-TF-IDF (class-based TF-IDF)
topics, probabilities = basic_topic_model.fit_transform(sample_documents)

print(f"Number of topics found: {len(set(topics)) - (1 if -1 in topics else 0)}")
print(f"Number of outliers: {list(topics).count(-1)}")

# Let's see what topics we discovered
print("\nTopics discovered:")
for topic_id, topic_words in basic_topic_model.get_topic_info().iterrows():
    if topic_id < 5:  # Show first 5 topics
        print(f"Topic {topic_words['Topic']}: {topic_words['Name']}")

# =============================================================================
# STEP 2: CUSTOMIZED BERTOPIC WITH FINE-TUNED COMPONENTS
# =============================================================================

print("\n=== CUSTOMIZED BERTOPIC IMPLEMENTATION ===")

# Now let's get more sophisticated and customize each component
# This gives us much more control over the process

# Step 2a: Choose our embedding model
# Different models work better for different types of text
# all-MiniLM-L6-v2 is fast and good for general purpose
# all-mpnet-base-v2 is slower but more accurate
# Use specialized models for domain-specific text (legal, medical, etc.)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Using sentence transformer for embeddings...")

# Step 2b: Configure UMAP for dimensionality reduction
# UMAP is great because it preserves both local and global structure
# n_neighbors: larger values = more global structure, smaller = more local
# n_components: dimensions to reduce to (usually 5-50 for clustering)
# min_dist: how tightly packed the embedding points can be
umap_model = UMAP(
    n_neighbors=15,      # Sweet spot for most text data
    n_components=5,      # 5D works well as input to HDBSCAN
    min_dist=0.0,       # Allow tight clusters
    metric='cosine',     # Cosine similarity works well for text embeddings
    random_state=42     # For reproducibility
)

# Step 2c: Configure HDBSCAN for clustering
# HDBSCAN is density-based clustering that can find clusters of varying shapes
# min_cluster_size: minimum number of documents per topic
# metric: distance metric (euclidean works well for UMAP output)
# cluster_selection_method: 'eom' is usually better than 'leaf'
hdbscan_model = HDBSCAN(
    min_cluster_size=10,                    # At least 10 docs per topic
    metric='euclidean',                     # Works well with UMAP
    cluster_selection_method='eom',         # Excess of Mass method
    prediction_data=True                    # Allows soft clustering
)

# Step 2d: Customize the vectorizer for topic representation
# This controls how topics are represented with words
# We can filter out common words, set n-gram ranges, etc.
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),          # Include both unigrams and bigrams
    stop_words="english",        # Remove common English stop words
    max_features=5000,           # Limit vocabulary size
    min_df=2,                   # Word must appear in at least 2 documents
    max_df=0.95                 # Word can't appear in more than 95% of docs
)

# Step 2e: Create our customized BERTopic model
# Now we combine all our custom components
custom_topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    top_k_words=10,             # Number of words per topic
    nr_topics="auto",           # Let the algorithm decide number of topics
    verbose=True
)

# Fit our custom model
print("Fitting customized BERTopic model...")
custom_topics, custom_probabilities = custom_topic_model.fit_transform(sample_documents)

print(f"Custom model found {len(set(custom_topics)) - (1 if -1 in custom_topics else 0)} topics")
print(f"Custom model outliers: {list(custom_topics).count(-1)}")

# =============================================================================
# STEP 3: ANALYZING AND INTERPRETING RESULTS
# =============================================================================

print("\n=== ANALYZING RESULTS ===")

# Get topic information - this is your main results summary
topic_info = custom_topic_model.get_topic_info()
print("\nTopic Information:")
print(topic_info.head(10))

# Look at individual topics in detail
print("\nDetailed Topic Analysis:")
for topic_id in range(min(5, len(set(custom_topics)) - (1 if -1 in custom_topics else 0))):
    print(f"\n--- Topic {topic_id} ---")
    topic_words = custom_topic_model.get_topic(topic_id)
    print(f"Top words: {[word for word, score in topic_words[:5]]}")
    
    # Get representative documents for this topic
    representative_docs = custom_topic_model.get_representative_docs(topic_id)
    if representative_docs:
        print(f"Representative document: {representative_docs[0][:100]}...")

# Create a document-topic dataframe for analysis
# This is super useful for understanding what your model learned
doc_topic_df = pd.DataFrame({
    'Document': sample_documents,
    'Topic': custom_topics,
    'Probability': custom_probabilities if custom_probabilities is not None else [0] * len(sample_documents)
})

print(f"\nDocument-Topic Distribution:")
print(doc_topic_df['Topic'].value_counts().head())

# =============================================================================
# STEP 4: ADVANCED TECHNIQUES AND OPTIMIZATIONS
# =============================================================================

print("\n=== ADVANCED TECHNIQUES ===")

# Technique 1: Topic Reduction
# Sometimes we get too many small topics - let's merge similar ones
print("Original number of topics:", len(set(custom_topics)) - (1 if -1 in custom_topics else 0))

# Reduce topics to a specific number (let's say 5)
custom_topic_model.reduce_topics(sample_documents, nr_topics=5)
reduced_topics = custom_topic_model.topics_

print("After reduction:", len(set(reduced_topics)) - (1 if -1 in reduced_topics else 0))

# Technique 2: Update topic representation
# We can use different methods to represent topics beyond c-TF-IDF
# Options include: KeyBERT, MMR (Maximal Marginal Relevance), POS tagging

# Let's try MMR to get more diverse words per topic
custom_topic_model.update_topics(
    sample_documents, 
    representation_model="mmr",  # Maximal Marginal Relevance
    top_k_words=15
)

print("\nUpdated topics with MMR:")
for topic_id in range(min(3, len(set(reduced_topics)) - (1 if -1 in reduced_topics else 0))):
    topic_words = custom_topic_model.get_topic(topic_id)
    print(f"Topic {topic_id}: {[word for word, score in topic_words[:5]]}")

# Technique 3: Hierarchical topic modeling
# This shows how topics relate to each other in a tree structure
hierarchical_topics = custom_topic_model.hierarchical_topics(sample_documents)
print(f"\nHierarchical topics shape: {hierarchical_topics.shape}")

# =============================================================================
# STEP 5: VISUALIZATION AND INTERPRETATION
# =============================================================================

print("\n=== CREATING VISUALIZATIONS ===")

# Unfortunately, we can't actually display plots in this code environment,
# but here's how you would create various visualizations

def create_visualizations(topic_model, docs, topics):
    """
    Create various visualizations for topic analysis
    This function shows all the different ways to visualize your results
    """
    
    # 1. Topic word scores - shows the importance of words in each topic
    # This is like a bar chart for each topic
    fig1 = topic_model.visualize_barchart(top_k_words=8)
    fig1.write_html("topic_barchart.html")
    print("Created topic bar chart visualization")
    
    # 2. Intertopic distance map - shows how topics relate to each other
    # Similar topics will be closer together
    fig2 = topic_model.visualize_topics()
    fig2.write_html("topic_distance_map.html")
    print("Created intertopic distance map")
    
    # 3. Topic hierarchy - shows the hierarchical relationship
    fig3 = topic_model.visualize_hierarchy()
    fig3.write_html("topic_hierarchy.html")
    print("Created topic hierarchy visualization")
    
    # 4. Topic over time (if you have timestamps)
    # This would show how topics evolve over time
    # timestamps = pd.date_range('2023-01-01', periods=len(docs), freq='D')
    # fig4 = topic_model.visualize_topics_over_time(topics_over_time, timestamps)
    
    # 5. Document similarity heatmap
    fig5 = topic_model.visualize_heatmap()
    fig5.write_html("topic_heatmap.html")
    print("Created topic similarity heatmap")
    
    return "Visualizations created successfully!"

# Create all visualizations
viz_result = create_visualizations(custom_topic_model, sample_documents, custom_topics)
print(viz_result)

# =============================================================================
# STEP 6: PRACTICAL TIPS AND BEST PRACTICES
# =============================================================================

print("\n=== PRACTICAL TIPS FOR REAL-WORLD USAGE ===")

def bertopic_best_practices():
    """
    Here are the key lessons I've learned from using BERTopic in production
    """
    
    tips = [
        "1. DATA PREPROCESSING:",
        "   - Clean your text but don't over-clean (keep meaningful punctuation)",
        "   - Remove duplicates - they can create artificial clusters",
        "   - Consider text length - very short texts are hard to cluster",
        "   - For social media data, keep hashtags and mentions - they're informative",
        "",
        "2. CHOOSING PARAMETERS:",
        "   - min_cluster_size: Start with sqrt(num_documents) / 2",
        "   - UMAP n_neighbors: 10-50 depending on dataset size",
        "   - For small datasets (<1000 docs), use smaller parameters",
        "   - For large datasets (>10000 docs), you can use larger parameters",
        "",
        "3. EMBEDDING MODELS:",
        "   - all-MiniLM-L6-v2: Fast and good for general text",
        "   - all-mpnet-base-v2: Better quality but slower",
        "   - Use domain-specific models for specialized text",
        "   - Consider multilingual models for non-English text",
        "",
        "4. EVALUATION AND ITERATION:",
        "   - Always manually inspect a sample of documents per topic",
        "   - Check if outlier documents (-1 topic) make sense",
        "   - Use topic coherence scores if you have ground truth",
        "   - Iterate on parameters based on domain knowledge",
        "",
        "5. PRODUCTION CONSIDERATIONS:",
        "   - Save your trained model: topic_model.save('my_model')",
        "   - For new documents: topic_model.transform(new_docs)",
        "   - Monitor topic drift over time in streaming data",
        "   - Consider computational resources for large datasets"
    ]
    
    return tips

tips = bertopic_best_practices()
for tip in tips:
    print(tip)

# =============================================================================
# STEP 7: SAVING AND LOADING MODELS
# =============================================================================

print("\n=== SAVING AND LOADING MODELS ===")

# Save the trained model for later use
# This is crucial for production systems
model_path = "bertopic_model"
custom_topic_model.save(model_path)
print(f"Model saved to {model_path}")

# Load the model (this is how you'd use it in production)
# loaded_model = BERTopic.load(model_path)
# new_topics, new_probs = loaded_model.transform(new_documents)

# =============================================================================
# STEP 8: EXAMPLE ANALYSIS WORKFLOW
# =============================================================================

def complete_analysis_workflow(documents, model_name="analysis_model"):
    """
    This is a complete workflow you can use as a template for your own projects
    It includes all the steps from data preprocessing to final analysis
    """
    
    print(f"\n=== COMPLETE ANALYSIS WORKFLOW ===")
    print(f"Analyzing {len(documents)} documents...")
    
    # Step 1: Initialize model with good defaults
    model = BERTopic(
        embedding_model=SentenceTransformer('all-MiniLM-L6-v2'),
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine'),
        hdbscan_model=HDBSCAN(min_cluster_size=max(10, len(documents)//50)),
        vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words="english"),
        verbose=True
    )
    
    # Step 2: Fit model
    topics, probs = model.fit_transform(documents)
    
    # Step 3: Basic analysis
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = list(topics).count(-1)
    
    print(f"Results: {n_topics} topics, {n_outliers} outliers")
    
    # Step 4: Topic quality check
    if n_topics > len(documents) // 20:  # Too many topics
        print("Too many topics detected, reducing...")
        model.reduce_topics(documents, nr_topics=max(5, len(documents)//40))
    
    # Step 5: Save results
    model.save(model_name)
    
    # Step 6: Create summary
    topic_info = model.get_topic_info()
    summary = {
        'n_topics': n_topics,
        'n_outliers': n_outliers,
        'largest_topic_size': topic_info['Count'].max() if not topic_info.empty else 0,
        'model_path': model_name
    }
    
    return model, summary

# Run the complete workflow on our sample data
final_model, analysis_summary = complete_analysis_workflow(sample_documents, "final_model")

print("\n=== FINAL ANALYSIS SUMMARY ===")
for key, value in analysis_summary.items():
    print(f"{key}: {value}")

print("\n=== CONCLUSION ===")
print("BERTopic is incredibly powerful for topic modeling because:")
print("- It combines state-of-the-art embeddings with proven clustering techniques")
print("- It's highly customizable while still working well out-of-the-box")
print("- It provides excellent visualizations for understanding results")
print("- It scales well from small experiments to production systems")
print("\nThe key to success is understanding your data and iterating on parameters!")
print("Always validate results manually and adjust based on domain knowledge.")
