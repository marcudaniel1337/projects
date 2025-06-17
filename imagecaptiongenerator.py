import os
import numpy as np
import string
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add

# ---------------- STEP 1: PREPROCESSING CAPTIONS ---------------- #

def load_and_clean_descriptions(filepath):
    table = str.maketrans('', '', string.punctuation)
    descriptions = {}
    with open(filepath, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            image_id, image_desc = tokens[0], tokens[1:]
            desc = ' '.join([w.lower().translate(table) for w in image_desc if w.isalpha()])
            caption = 'startseq ' + desc + ' endseq'
            if image_id not in descriptions:
                descriptions[image_id] = []
            descriptions[image_id].append(caption)
    return descriptions

# ---------------- STEP 2: EXTRACT IMAGE FEATURES ---------------- #

def extract_image_features(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(model.input, model.layers[-2].output)
    features = {}
    for img_name in tqdm(os.listdir(directory)):
        path = os.path.join(directory, img_name)
        img = Image.open(path).resize((299, 299)).convert('RGB')
        img = np.expand_dims(np.array(img), axis=0)
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        features[img_name] = feature
    return features

# ---------------- STEP 3: TOKENIZE CAPTIONS ---------------- #

def create_tokenizer(descriptions):
    lines = [cap for descs in descriptions.values() for cap in descs]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def get_max_length(descriptions):
    lines = [cap for descs in descriptions.values() for cap in descs]
    return max(len(d.split()) for d in lines)

# ---------------- STEP 4: DEFINE ATTENTION COMPONENT ---------------- #

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# ---------------- STEP 5: DECODER WITH ATTENTION ---------------- #

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super().__init__()
        self.units = units
        self.attention = BahdanauAttention(units)
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(units, return_sequences=True, return_state=True)
        self.fc1 = Dense(units)
        self.fc2 = Dense(vocab_size)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state, _ = self.lstm(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

# ---------------- STEP 6: DATA GENERATOR ---------------- #

def data_generator(descriptions, photos, tokenizer, max_len, vocab_size):
    while True:
        for img_id, desc_list in descriptions.items():
            photo = photos[img_id][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield [[photo, in_seq], out_seq]

# ---------------- STEP 7: GENERATE CAPTIONS (BEAM SEARCH) ---------------- #

def generate_caption_beam_search(photo, model, tokenizer, max_len, beam_index=3):
    start_seq = [tokenizer.word_index['startseq']]
    sequences = [[start_seq, 0.0]]
    
    while len(sequences[0][0]) < max_len:
        temp = []
        for s in sequences:
            seq = pad_sequences([s[0]], maxlen=max_len)
            preds = model.predict([photo, seq], verbose=0)[0]
            top_preds = np.argsort(preds)[-beam_index:]
            for word in top_preds:
                new_seq = s[0] + [word]
                new_score = s[1] + np.log(preds[word])
                temp.append([new_seq, new_score])
        sequences = sorted(temp, key=lambda x: x[1], reverse=True)[:beam_index]

    final_words = [tokenizer.index_word[i] for i in sequences[0][0] if i > 0 and i != tokenizer.word_index['endseq']]
    return ' '.join(final_words)

# ---------------- STEP 8: PLOT ATTENTION ---------------- #

def plot_attention(image, result, attention_plot):
    fig = plt.figure(figsize=(10, 10))
    for l in range(len(result)):
        temp_att = attention_plot[l].reshape((8, 8))
        ax = fig.add_subplot(np.ceil(len(result)/2.), 2, l+1)
        ax.set_title(result[l])
        ax.imshow(image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ---------------- USAGE EXAMPLE ---------------- #

"""
descriptions = load_and_clean_descriptions('Flickr8k.token.txt')
features = extract_image_features('Flickr8k_Dataset/Images')
tokenizer = create_tokenizer(descriptions)
max_len = get_max_length(descriptions)
vocab_size = len(tokenizer.word_index) + 1

decoder = Decoder(vocab_size=vocab_size, embedding_dim=256, units=256)
# You would then compile and fit the decoder with a training loop using the generator

# For inference:
# caption = generate_caption_beam_search(photo, decoder, tokenizer, max_len)
"""
