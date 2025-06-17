import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pickle
import os

# Let's start by defining our vocabulary class - think of this as our dictionary
# that converts words to numbers and vice versa (computers love numbers!)
class Vocabulary:
    def __init__(self):
        # These special tokens help our model understand sentence structure
        self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}  # Start of sentence, end of sentence, unknown word
        self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.word_count = {}
        self.n_words = 4  # Count of words (starting with our 4 special tokens)
    
    def add_sentence(self, sentence):
        """Add all words from a sentence to our vocabulary - like building a personal dictionary"""
        # Clean and split the sentence into words
        words = self.clean_text(sentence).split()
        for word in words:
            self.add_word(word)
    
    def add_word(self, word):
        """Add a single word to vocabulary - each new word gets a unique number"""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
    
    def clean_text(self, text):
        """Clean text by removing weird characters - like proofreading before publishing"""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Keep only letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
        return text
    
    def sentence_to_indices(self, sentence, max_length=None):
        """Convert a sentence to a list of numbers our model can understand"""
        words = self.clean_text(sentence).split()
        indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in words]
        
        # Truncate if too long (like editing a long essay to fit page limits)
        if max_length and len(indices) > max_length - 1:
            indices = indices[:max_length - 1]
        
        # Add end-of-sentence marker
        indices.append(self.word2idx['<eos>'])
        return indices
    
    def indices_to_sentence(self, indices):
        """Convert numbers back to readable text - like decoding a secret message"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<eos>']:
                break  # Stop at end of sentence
            if idx == self.word2idx['<pad>']:
                continue  # Skip padding tokens
            words.append(self.idx2word.get(idx, '<unk>'))
        return ' '.join(words)

# Our dataset class - think of this as organizing our training materials
class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, vocab, max_article_len=512, max_summary_len=128):
        """
        articles: list of full text articles (like newspaper articles)
        summaries: list of corresponding summaries (like headlines or abstracts)
        vocab: our vocabulary object
        max_*_len: maximum length limits (like word limits for essays)
        """
        self.articles = articles
        self.summaries = summaries
        self.vocab = vocab
        self.max_article_len = max_article_len
        self.max_summary_len = max_summary_len
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        """Get one training example - like picking one article-summary pair from our pile"""
        # Convert text to numbers
        article_indices = self.vocab.sentence_to_indices(self.articles[idx], self.max_article_len)
        summary_indices = self.vocab.sentence_to_indices(self.summaries[idx], self.max_summary_len)
        
        # Pad sequences to same length (like making sure all pages have the same margins)
        article_padded = self.pad_sequence(article_indices, self.max_article_len)
        summary_padded = self.pad_sequence(summary_indices, self.max_summary_len)
        
        return {
            'article': torch.tensor(article_padded, dtype=torch.long),
            'summary': torch.tensor(summary_padded, dtype=torch.long),
            'article_length': len(article_indices),
            'summary_length': len(summary_indices)
        }
    
    def pad_sequence(self, sequence, max_length):
        """Add padding to make all sequences the same length - like adding blank lines to align text"""
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.vocab.word2idx['<pad>']] * (max_length - len(sequence))

# Positional Encoding - helps the model understand word order (like page numbers in a book)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model: dimension of our word embeddings (like how many features describe each word)
        max_len: maximum sequence length we'll handle
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to hold position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Use sine and cosine functions to create unique position signatures
        # Think of this like creating a unique barcode for each position
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even positions get sine
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd positions get cosine
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # Reshape for batch processing
        self.register_buffer('pe', pe)  # Save as part of model but don't train it
    
    def forward(self, x):
        """Add position information to word embeddings"""
        # x shape: (sequence_length, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

# Multi-Head Attention - the "brain" of the transformer that decides what to focus on
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        d_model: dimension of our embeddings (like 512)
        n_heads: number of attention heads (like having multiple experts look at the same thing)
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0  # Make sure we can split evenly
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear layers to transform inputs - like different "lenses" to view the data
        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation  
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        The core attention mechanism - like deciding which words to pay attention to
        Q: queries (what we're looking for)
        K: keys (what we're looking at)  
        V: values (the actual information)
        """
        # Calculate attention scores - how much should we focus on each word?
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (like covering up words we shouldn't look at)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Convert scores to probabilities - like deciding percentages of attention
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values - like highlighting important parts of text
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Transform inputs through linear layers and split into multiple heads
        # Like having multiple people read the same text with different perspectives
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention mechanism
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate all heads back together - like combining different perspectives
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights

# Feed Forward Network - processes information after attention (like digesting what we learned)
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model: input/output dimension
        d_ff: hidden dimension (usually 4 times larger than d_model)
        """
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Expand, activate, then compress - like thinking deeply then summarizing
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# Encoder Layer - one complete "reading comprehension" step
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)  # Normalization for stable training
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection - like taking notes while reading
        attention_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed forward with residual connection - like processing those notes
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Decoder Layer - one complete "writing" step that looks at both input and previous output
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention on target (what we've written so far)
        self_attention_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attention_output))
        
        # Cross-attention with encoder output (looking back at the source text)
        encoder_attention_output, _ = self.encoder_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(encoder_attention_output))
        
        # Feed forward processing
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

# Complete Transformer Model - our AI summarizer!
class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, 
                 max_seq_len=512, dropout=0.1):
        """
        vocab_size: size of our vocabulary (how many different words we know)
        d_model: dimension of embeddings (richness of word representation)
        n_heads: number of attention heads (multiple perspectives)
        n_layers: number of transformer layers (depth of processing)
        d_ff: feed-forward dimension (thinking capacity)
        max_seq_len: maximum sequence length
        dropout: regularization rate (prevents overfitting)
        """
        super(TransformerSummarizer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Word embeddings - convert word indices to rich vector representations
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Stack of encoder layers - like multiple rounds of reading comprehension
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # Stack of decoder layers - like multiple rounds of writing and editing
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # Final layer to convert back to vocabulary probabilities
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def create_padding_mask(self, seq, pad_token=0):
        """Create mask to ignore padding tokens - like ignoring blank spaces"""
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        """Create mask to prevent looking at future tokens during training"""
        # Like covering up the rest of the sentence while writing each word
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def encode(self, src, src_mask=None):
        """Encode the source text - like reading and understanding the article"""
        # Convert word indices to embeddings and add position information
        src_embedded = self.embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        src_embedded = self.dropout(src_embedded)
        
        # Pass through all encoder layers
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
            
        return encoder_output
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode to generate summary - like writing the summary word by word"""
        # Convert target indices to embeddings and add position information
        tgt_embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        tgt_embedded = self.dropout(tgt_embedded)
        
        # Pass through all decoder layers
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, src_mask, tgt_mask)
            
        return decoder_output
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """Complete forward pass - read article and generate summary"""
        # Encode the source text
        encoder_output = self.encode(src, src_mask)
        
        # Decode to generate summary
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Convert to vocabulary probabilities
        output = self.output_projection(decoder_output)
        
        return output

# Training function - teaches our model to summarize
def train_model(model, dataloader, vocab, num_epochs=10, learning_rate=0.0001):
    """
    Train the summarization model
    model: our transformer model
    dataloader: provides training data in batches
    vocab: vocabulary object
    num_epochs: how many times to go through all training data
    learning_rate: how fast the model learns
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer - decides how to update model weights (like a learning strategy)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # Loss function - measures how wrong our predictions are
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
    
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to GPU if available
            src = batch['article'].to(device)  # Source articles
            tgt = batch['summary'].to(device)  # Target summaries
            
            # Prepare target input and output
            # Target input: <sos> + summary (what we feed to decoder)
            # Target output: summary + <eos> (what we want to predict)
            tgt_input = tgt[:, :-1]  # All but last token
            tgt_output = tgt[:, 1:]  # All but first token
            
            # Create masks
            src_mask = model.create_padding_mask(src, vocab.word2idx['<pad>']).to(device)
            tgt_mask = model.create_padding_mask(tgt_input, vocab.word2idx['<pad>']).to(device)
            
            # Create look-ahead mask for target
            seq_len = tgt_input.size(1)
            look_ahead_mask = model.create_look_ahead_mask(seq_len).to(device)
            tgt_mask = tgt_mask & look_ahead_mask
            
            # Forward pass
            optimizer.zero_grad()  # Reset gradients
            outputs = model(src, tgt_input, src_mask, tgt_mask)
            
            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}, Average Loss: {avg_loss:.4f}")
        
        # Print epoch summary
        avg_epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}\n")

# Inference function - use trained model to generate summaries
def generate_summary(model, article_text, vocab, max_length=100, device=None):
    """
    Generate a summary for a given article
    model: trained transformer model
    article_text: the article to summarize (string)
    vocab: vocabulary object
    max_length: maximum length of generated summary
    device: GPU or CPU
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():  # Don't calculate gradients during inference
        # Prepare input
        src_indices = vocab.sentence_to_indices(article_text, 512)
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
        
        # Create source mask
        src_mask = model.create_padding_mask(src_tensor, vocab.word2idx['<pad>']).to(device)
        
        # Encode the source
        encoder_output = model.encode(src_tensor, src_mask)
        
        # Start with <sos> token
        decoder_input = torch.tensor([[vocab.word2idx['<sos>']]], dtype=torch.long).to(device)
        
        # Generate summary word by word
        for _ in range(max_length):
            # Create target mask
            tgt_mask = model.create_padding_mask(decoder_input, vocab.word2idx['<pad>']).to(device)
            seq_len = decoder_input.size(1)
            look_ahead_mask = model.create_look_ahead_mask(seq_len).to(device)
            tgt_mask = tgt_mask & look_ahead_mask
            
            # Decode next token
            decoder_output = model.decode(decoder_input, encoder_output, src_mask, tgt_mask)
            next_token_logits = model.output_projection(decoder_output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Stop if we generate <eos> token
            if next_token.item() == vocab.word2idx['<eos>']:
                break
                
            # Add predicted token to input for next iteration
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # Convert indices back to text
        summary_indices = decoder_input.squeeze().cpu().numpy().tolist()
        summary_text = vocab.indices_to_sentence(summary_indices[1:])  # Skip <sos> token
        
        return summary_text

# Example usage and training setup
def main():
    """Main function to demonstrate the complete summarization pipeline"""
    
    # Sample data - in real use, you'd load from files or datasets
    sample_articles = [
        "The recent breakthrough in artificial intelligence has revolutionized the way we approach complex problems. Researchers at leading universities have developed new algorithms that can process vast amounts of data in seconds. This technology promises to transform industries ranging from healthcare to finance. The implications are far-reaching and could change how we work and live.",
        "Climate change continues to be one of the most pressing issues of our time. Rising global temperatures are causing ice caps to melt at an unprecedented rate. Scientists warn that immediate action is needed to prevent catastrophic consequences. Governments worldwide are implementing new policies to reduce carbon emissions and promote sustainable energy sources.",
        "The stock market experienced significant volatility this week due to geopolitical tensions and economic uncertainty. Investors are closely monitoring inflation rates and central bank decisions. Technology stocks showed mixed performance while energy sectors gained momentum. Financial experts recommend diversifying portfolios during these uncertain times."
    ]
    
    sample_summaries = [
        "AI breakthrough develops new algorithms that process data quickly, transforming multiple industries.",
        "Climate change causes rapid ice cap melting, requiring immediate global action and sustainable policies.",
        "Stock market volatility from geopolitical tensions affects various sectors, with experts advising portfolio diversification."
    ]
    
    print("Building vocabulary from sample data...")
    # Build vocabulary
    vocab = Vocabulary()
    for article in sample_articles:
        vocab.add_sentence(article)
    for summary in sample_summaries:
        vocab.add_sentence(summary)
    
    print(f"Vocabulary size: {vocab.n_words}")
    
    # Create dataset and dataloader
    dataset = SummarizationDataset(sample_articles, sample_summaries, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = TransformerSummarizer(
        vocab_size=vocab.n_words,
        d_model=256,  # Smaller model for demonstration
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train the model (in real scenarios, you'd train for many more epochs)
    print("Starting training...")
    train_model(model, dataloader, vocab, num_epochs=5, learning_rate=0.001)
    
    # Test the model
    print("\nTesting the model...")
    test_article = "Artificial intelligence and machine learning technologies are advancing rapidly. These developments are creating new opportunities in various fields including medicine, transportation, and communication. However, there are also concerns about job displacement and privacy issues that need to be addressed."
    
    summary = generate_summary(model, test_article, vocab, max_length=50)
    print(f"Original article: {test_article}")
    print(f"Generated summary: {summary}")
    
    # Save the trained model and vocabulary
    print("\nSaving model and vocabulary...")
    torch.save(model.state_dict(), 'summarizer_model.pth')
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Model and vocabulary saved successfully!")

if __name__ == "__main__":
    main()
