"""
Semantic Search Engine using BERT and FAISS

This implementation creates a semantic search system that understands the meaning
behind queries rather than just matching keywords. Think of it like having a 
librarian who actually understands what you're asking for, not just someone
who matches the exact words you said.

Key components:
- BERT: Converts text into numerical representations (embeddings) that capture meaning
- FAISS: Lightning-fast similarity search through millions of embeddings
- Preprocessing: Cleans and prepares text for better understanding
"""

import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Optional
import re
import pickle
import logging
from dataclasses import dataclass
from pathlib import Path

# Set up logging so we can see what's happening under the hood
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """
    A container for search results that makes it easy to work with.
    Think of this as a neat package that holds everything we want to know
    about a search match.
    """
    text: str
    score: float  # Higher scores mean better matches
    index: int    # Position in the original dataset

class SemanticSearchEngine:
    """
    The main search engine class. This is where all the magic happens.
    
    The basic idea is:
    1. Take a bunch of documents and convert them to BERT embeddings
    2. Store these embeddings in FAISS for super-fast searching  
    3. When someone searches, convert their query to an embedding too
    4. Find the most similar document embeddings using cosine similarity
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the search engine with a pre-trained BERT model.
        
        Why this specific model? It's a good balance of speed and quality:
        - Small enough to run on most machines (22MB)
        - Trained specifically for sentence similarity tasks
        - Good performance across many domains
        
        Args:
            model_name: The name of the Hugging Face model to use
        """
        logger.info(f"Loading BERT model: {model_name}")
        
        # Load the tokenizer (converts text to tokens BERT understands)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the actual BERT model
        self.model = AutoModel.from_pretrained(model_name)
        
        # Put model in evaluation mode (turns off dropout, etc.)
        # This gives us consistent embeddings every time
        self.model.eval()
        
        # Use GPU if available, otherwise fall back to CPU
        # GPU makes things much faster, but CPU works fine for smaller datasets
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # These will be set when we build the index
        self.index = None
        self.documents = []  # Keep original texts for returning results
        self.embeddings = None  # Store embeddings for potential reuse
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean up text before feeding it to BERT.
        
        Why preprocess? BERT is powerful but it works better with clean text:
        - Extra whitespace can confuse tokenization
        - Very long texts might get truncated inconsistently
        - Some characters might not be handled well
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text ready for BERT
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Replace multiple whitespace with single space
        # "Hello    world\n\n" becomes "Hello world"
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Truncate if too long (BERT has a 512 token limit)
        # We use 500 to be safe and leave room for special tokens
        if len(text) > 500:
            text = text[:500]
            logger.warning("Text truncated to fit BERT's token limit")
        
        return text
    
    def _get_bert_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert a list of texts into BERT embeddings.
        
        This is the heart of semantic search - converting human language into
        numerical vectors that capture meaning. Similar texts will have similar
        vectors, which lets us do mathematical similarity comparisons.
        
        Args:
            texts: List of texts to convert
            batch_size: How many texts to process at once (larger = faster but more memory)
            
        Returns:
            Array of embeddings, shape (num_texts, embedding_dimension)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        embeddings = []
        
        # Process in batches to avoid running out of memory
        # Think of this like doing laundry - you don't put all clothes in at once
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch (convert text to numbers BERT understands)
            inputs = self.tokenizer(
                batch_texts,
                padding=True,      # Pad shorter texts so all have same length
                truncation=True,   # Cut off texts that are too long
                return_tensors='pt',  # Return PyTorch tensors
                max_length=512     # BERT's maximum sequence length
            )
            
            # Move to GPU if available
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings without computing gradients (saves memory)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get the embeddings from the last hidden layer
                # We use mean pooling to convert from token-level to sentence-level
                # Think of this as averaging all word meanings to get sentence meaning
                last_hidden_states = outputs.last_hidden_state
                
                # Create attention mask to ignore padding tokens in mean calculation
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                
                # Mean pooling: sum all token embeddings, then divide by number of real tokens
                masked_embeddings = last_hidden_states * attention_mask
                summed = torch.sum(masked_embeddings, dim=1)
                counts = torch.sum(attention_mask, dim=1)
                mean_embeddings = summed / counts
                
                # Convert to numpy and store
                embeddings.append(mean_embeddings.cpu().numpy())
        
        # Combine all batches into one big array
        embeddings = np.vstack(embeddings)
        
        # Normalize vectors so we can use dot product for cosine similarity
        # This makes all vectors have length 1, so similarity = dot product
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_index(self, documents: List[str], save_path: Optional[str] = None):
        """
        Build the FAISS search index from a collection of documents.
        
        This is like creating a super-efficient phone book for embeddings.
        Instead of looking through every document one by one, FAISS creates
        a data structure that lets us jump straight to the most similar ones.
        
        Args:
            documents: List of text documents to index
            save_path: Optional path to save the index for later use
        """
        logger.info(f"Building search index for {len(documents)} documents...")
        
        # Clean up all documents
        processed_docs = [self._preprocess_text(doc) for doc in documents]
        
        # Remove empty documents (they would cause problems)
        valid_docs = [(i, doc) for i, doc in enumerate(processed_docs) if doc.strip()]
        
        if len(valid_docs) < len(documents):
            logger.warning(f"Removed {len(documents) - len(valid_docs)} empty documents")
        
        # Store the mapping back to original indices
        self.original_indices = [i for i, _ in valid_docs]
        self.documents = [doc for _, doc in valid_docs]
        
        # Generate BERT embeddings
        self.embeddings = self._get_bert_embeddings(self.documents)
        
        # Create FAISS index
        # We use IndexFlatIP (Inner Product) because our embeddings are normalized
        # so inner product = cosine similarity
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Add all embeddings to the index
        # FAISS expects float32, and our embeddings might be float64
        self.index.add(self.embeddings.astype(np.float32))
        
        logger.info(f"Index built successfully with {self.index.ntotal} documents")
        
        # Save everything if path provided
        if save_path:
            self.save_index(save_path)
    
    def save_index(self, save_path: str):
        """
        Save the search index to disk so we don't have to rebuild it every time.
        
        Building indices can take a while for large document collections,
        so saving them is like bookmarking your work.
        
        Args:
            save_path: Directory path where to save the index files
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index (the fast lookup structure)
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        
        # Save other data we need (documents, embeddings, mappings)
        with open(save_path / "metadata.pkl", "wb") as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'original_indices': self.original_indices
            }, f)
        
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, save_path: str):
        """
        Load a previously saved index from disk.
        
        This lets us skip the time-consuming index building step
        and jump straight to searching.
        
        Args:
            save_path: Directory path where the index files are stored
        """
        save_path = Path(save_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(save_path / "faiss.index"))
        
        # Load metadata
        with open(save_path / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.original_indices = data['original_indices']
        
        logger.info(f"Index loaded from {save_path}")
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        """
        Search for documents similar to the query.
        
        This is where the magic happens - we convert the user's query into
        an embedding and find the most similar document embeddings.
        
        Args:
            query: The search query (natural language)
            top_k: How many results to return
            score_threshold: Minimum similarity score (0-1, higher = more similar)
            
        Returns:
            List of SearchResult objects, sorted by similarity score
        """
        if self.index is None:
            raise ValueError("No index found. Please build or load an index first.")
        
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Clean and convert query to embedding
        processed_query = self._preprocess_text(query)
        query_embedding = self._get_bert_embeddings([processed_query])
        
        # Search the index
        # FAISS returns similarity scores and indices of the most similar documents
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Convert results to our SearchResult format
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Skip invalid indices (can happen if we ask for more results than exist)
            if idx == -1:
                continue
            
            # Skip results below our threshold
            if score < score_threshold:
                continue
            
            results.append(SearchResult(
                text=self.documents[idx],
                score=float(score),
                index=self.original_indices[idx]
            ))
        
        logger.info(f"Found {len(results)} results for query: '{query[:50]}...'")
        return results
    
    def get_similar_documents(self, doc_index: int, top_k: int = 5) -> List[SearchResult]:
        """
        Find documents similar to a specific document in the index.
        
        This is useful for "more like this" functionality - show me other
        documents similar to this one the user is currently viewing.
        
        Args:
            doc_index: Index of the document to find similarities for
            top_k: How many similar documents to return
            
        Returns:
            List of similar documents (excluding the original)
        """
        if self.index is None:
            raise ValueError("No index found. Please build or load an index first.")
        
        if doc_index >= len(self.documents):
            raise ValueError(f"Document index {doc_index} out of range")
        
        # Get the embedding for this document
        doc_embedding = self.embeddings[doc_index:doc_index+1]
        
        # Search for similar documents (we get top_k+1 because the first result will be the document itself)
        scores, indices = self.index.search(doc_embedding.astype(np.float32), top_k + 1)
        
        # Convert to results, skipping the first one (which is the document itself)
        results = []
        for score, idx in zip(scores[0][1:], indices[0][1:]):  # Skip first result
            if idx == -1:
                continue
                
            results.append(SearchResult(
                text=self.documents[idx],
                score=float(score),
                index=self.original_indices[idx]
            ))
        
        return results

def demo_usage():
    """
    A demonstration of how to use the semantic search engine.
    
    This shows the typical workflow:
    1. Create sample documents
    2. Build search index
    3. Perform searches
    4. Save/load indices
    """
    
    # Sample documents for demonstration
    # Notice how these cover similar topics but use different words
    sample_docs = [
        "The quick brown fox jumps over the lazy dog in the forest.",
        "Machine learning algorithms can recognize patterns in large datasets.",
        "Python is a popular programming language for data science and AI.",
        "Dogs are loyal companions and make great pets for families.",
        "Neural networks are inspired by the structure of the human brain.",
        "Cats are independent animals that require less attention than dogs.",
        "Deep learning has revolutionized computer vision and natural language processing.",
        "The weather today is sunny with a chance of rain in the afternoon.",
        "Artificial intelligence is transforming industries across the globe.",
        "Cooking is both an art and a science that brings people together."
    ]
    
    print("🔍 Semantic Search Engine Demo")
    print("=" * 50)
    
    # Initialize the search engine
    engine = SemanticSearchEngine()
    
    # Build the search index
    print("\n📚 Building search index...")
    engine.build_index(sample_docs)
    
    # Perform some example searches
    test_queries = [
        "artificial intelligence and machine learning",  # Should match AI/ML docs
        "pets and animals",                             # Should match pet-related docs  
        "programming and coding",                       # Should match programming docs
        "weather forecast"                              # Should match weather doc
    ]
    
    for query in test_queries:
        print(f"\n🔎 Searching for: '{query}'")
        print("-" * 40)
        
        results = engine.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.3f}")
            print(f"   Text: {result.text[:80]}...")
            print()
    
    # Demonstrate "find similar" functionality
    print("\n🔄 Finding documents similar to document #1:")
    print("-" * 50)
    similar_docs = engine.get_similar_documents(1, top_k=3)
    
    print(f"Original document: {sample_docs[1]}")
    print("\nSimilar documents:")
    for i, result in enumerate(similar_docs, 1):
        print(f"{i}. Score: {result.score:.3f}")
        print(f"   Text: {result.text}")
        print()
    
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    # Run the demonstration
    demo_usage()
