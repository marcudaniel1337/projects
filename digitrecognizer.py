import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

class DigitRecognizer:
    """
    A sophisticated digit recognition system that can learn to identify handwritten digits (0-9).
    
    Think of this as teaching a computer to recognize numbers the same way a child learns -
    by showing it thousands of examples until it recognizes patterns.
    """
    
    def __init__(self):
        # These will store our data and models - like setting up our workspace
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()  # This helps normalize our data - like adjusting brightness/contrast
        
    def load_data(self):
        """
        Load the famous MNIST dataset - it's like the "Hello World" of machine learning.
        Contains 70,000 handwritten digits from high school students and Census Bureau employees.
        
        Each image is 28x28 pixels, which means 784 individual pixel values per digit.
        It's amazing how much information is packed into such a small image!
        """
        print("📦 Loading MNIST dataset... (this might take a moment)")
        print("   Think of this as gathering thousands of handwriting samples for study")
        
        # Fetch the dataset - OpenML is like a library of datasets
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Let's see what we're working with
        print(f"   📊 Dataset shape: {X.shape} - that's {X.shape[0]:,} images!")
        print(f"   🎯 Labels shape: {y.shape}")
        print(f"   📏 Each image: 28x28 pixels = {X.shape[1]} features")
        
        # Split the data - like dividing flashcards into practice and test piles
        # We use 80% for training (learning) and 20% for testing (final exam)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   ✅ Training set: {self.X_train.shape[0]:,} images")
        print(f"   ✅ Test set: {self.X_test.shape[0]:,} images")
        
        # Normalize pixel values from 0-255 to 0-1
        # This is like adjusting the brightness so all images have similar contrast
        # Neural networks learn better when input values are in a smaller range
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        
        print("   🎨 Pixel values normalized (0-255 → 0-1)")
        
    def visualize_samples(self, num_samples=10):
        """
        Let's take a peek at what we're working with!
        This is like looking at a few handwriting samples before we start teaching.
        """
        if self.X_train is None:
            print("❌ No data loaded yet! Call load_data() first.")
            return
            
        print(f"🔍 Showing {num_samples} random samples from our training data...")
        
        # Create a nice grid to display the images
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        fig.suptitle('Sample Handwritten Digits from Training Data', fontsize=16)
        
        # Pick random samples to display
        random_indices = np.random.choice(len(self.X_train), num_samples, replace=False)
        
        for i, idx in enumerate(random_indices):
            row, col = i // 5, i % 5
            
            # Reshape the flat array back to 28x28 for display
            digit_image = self.X_train[idx].reshape(28, 28)
            
            axes[row, col].imshow(digit_image, cmap='gray')
            axes[row, col].set_title(f'Label: {self.y_train[idx]}', fontsize=12)
            axes[row, col].axis('off')  # Hide axis for cleaner look
            
        plt.tight_layout()
        plt.show()
        
        # Show the distribution of digits in our dataset
        self._show_digit_distribution()
        
    def _show_digit_distribution(self):
        """
        Check if our dataset is balanced - do we have roughly equal numbers of each digit?
        This is important because we don't want our model to be biased toward certain digits.
        """
        print("\n📊 Analyzing digit distribution in training data...")
        
        unique, counts = np.unique(self.y_train, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, color='skyblue', alpha=0.7)
        plt.xlabel('Digit')
        plt.ylabel('Count')
        plt.title('Distribution of Digits in Training Set')
        plt.xticks(unique)
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{count:,}', ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        
        print("   ✅ Dataset appears well-balanced - good for training!")
        
    def build_model(self, model_type='cnn'):
        """
        Build our neural network model. We have two options:
        
        1. Simple Dense Network: Like a basic brain with fully connected neurons
        2. CNN (Convolutional): Like a more sophisticated visual system that recognizes patterns
        
        CNNs are better for images because they can detect features like edges, curves, etc.
        """
        print(f"🧠 Building {model_type.upper()} model...")
        
        if model_type == 'cnn':
            self.model = self._build_cnn_model()
        else:
            self.model = self._build_dense_model()
            
        # Show model architecture - like looking at the blueprint
        print("\n📋 Model Architecture:")
        self.model.summary()
        
    def _build_cnn_model(self):
        """
        Build a Convolutional Neural Network - the gold standard for image recognition.
        
        Think of this as creating a visual system that first looks for simple patterns
        (edges, lines) then combines them into more complex features (loops, curves)
        and finally makes a decision about which digit it's seeing.
        """
        model = Sequential([
            # First, reshape our flat data back into 28x28 images
            # It's like unfolding a crumpled photo to see it properly
            Dense(784, input_shape=(784,), activation='relu'),
            
            # Reshape to image format for CNN layers
            # This is the magic moment where flat data becomes a 2D image again
            Lambda(lambda x: tf.reshape(x, (-1, 28, 28, 1))),
            
            # First convolutional layer - like giving the model "eyes"
            # 32 filters means it's looking for 32 different types of patterns
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            # Batch normalization helps training stability - like taking steady breaths
            BatchNormalization(),
            # Max pooling reduces image size while keeping important features
            MaxPooling2D((2, 2)),
            
            # Second conv layer - now it can recognize more complex patterns
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Third conv layer - even more sophisticated pattern recognition
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Flatten the 2D features into 1D for the final decision layers
            Flatten(),
            
            # Dense layers for final classification
            Dense(512, activation='relu'),
            Dropout(0.5),  # Dropout prevents overfitting - like teaching with variety
            
            Dense(256, activation='relu'),
            Dropout(0.3),
            
            # Final layer: 10 neurons for 10 digits (0-9)
            # Softmax gives us probabilities - "I'm 95% sure this is a 7"
            Dense(10, activation='softmax')
        ])
        
        return model
    
    def _build_dense_model(self):
        """
        Build a simpler fully-connected network.
        Less sophisticated than CNN but still effective and faster to train.
        """
        model = Sequential([
            # Input layer
            Dense(512, input_shape=(784,), activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(10, activation='softmax')
        ])
        
        return model
        
    def train_model(self, epochs=50, batch_size=128):
        """
        Train our model - this is where the actual learning happens!
        
        Think of epochs as "study sessions" - each epoch, the model sees
        all the training data once and adjusts its understanding.
        
        Batch size is like studying in groups - processing multiple examples
        at once is more efficient than one at a time.
        """
        if self.model is None:
            print("❌ No model built yet! Call build_model() first.")
            return
            
        print(f"🎓 Starting training for {epochs} epochs...")
        print(f"   📚 Batch size: {batch_size} (processing {batch_size} images at a time)")
        
        # Prepare labels for multi-class classification
        # Convert labels to one-hot encoding: 3 becomes [0,0,0,1,0,0,0,0,0,0]
        y_train_categorical = to_categorical(self.y_train, 10)
        y_test_categorical = to_categorical(self.y_test, 10)
        
        # Compile the model - set up the learning process
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),  # How fast the model learns
            loss='categorical_crossentropy',       # How we measure mistakes
            metrics=['accuracy']                   # What we track during training
        )
        
        # Set up callbacks - these help training be more intelligent
        callbacks = [
            # Stop early if we're not improving - prevents wasted time
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate if we plateau - like studying more carefully
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        print("🚀 Training in progress...")
        
        # The actual training happens here!
        history = self.model.fit(
            self.X_train, y_train_categorical,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_test, y_test_categorical),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        
        # Visualize the training progress
        self._plot_training_history(history)
        
        return history
    
    def _plot_training_history(self, history):
        """
        Visualize how well our model learned over time.
        This is like looking at progress reports throughout the semester.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy over time
        ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        ax1.set_title('Model Accuracy Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss over time
        ax2.plot(history.history['loss'], label='Training Loss', color='blue')
        ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
        ax2.set_title('Model Loss Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def evaluate_model(self):
        """
        Test our trained model on data it has never seen before.
        This is the moment of truth - how well did our "student" actually learn?
        """
        if self.model is None:
            print("❌ No model to evaluate! Train a model first.")
            return
            
        print("📊 Evaluating model performance...")
        
        # Make predictions on test data
        predictions = self.model.predict(self.X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, predicted_classes)
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show detailed classification report
        print("\n📋 Detailed Performance Report:")
        print(classification_report(self.y_test, predicted_classes))
        
        # Create confusion matrix - shows which digits get confused with which
        self._plot_confusion_matrix(self.y_test, predicted_classes)
        
        # Show some examples of correct and incorrect predictions
        self._show_prediction_examples(predicted_classes)
        
    def _plot_confusion_matrix(self, y_true, y_pred):
        """
        Create a confusion matrix - this shows us which digits our model
        gets confused between. For example, does it mix up 6s and 8s?
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix - Which Digits Get Mixed Up?')
        plt.xlabel('Predicted Digit')
        plt.ylabel('True Digit')
        plt.show()
        
        # Analyze the most common mistakes
        print("\n🔍 Most common mistakes:")
        mistakes = []
        for i in range(10):
            for j in range(10):
                if i != j and cm[i][j] > 10:  # More than 10 mistakes
                    mistakes.append((i, j, cm[i][j]))
        
        mistakes.sort(key=lambda x: x[2], reverse=True)
        for true_digit, pred_digit, count in mistakes[:5]:
            print(f"   {true_digit} → {pred_digit}: {count} times")
    
    def _show_prediction_examples(self, predicted_classes):
        """
        Show some examples of what our model got right and wrong.
        This helps us understand the model's strengths and weaknesses.
        """
        # Find some correct and incorrect predictions
        correct_mask = (predicted_classes == self.y_test)
        incorrect_mask = ~correct_mask
        
        correct_indices = np.where(correct_mask)[0]
        incorrect_indices = np.where(incorrect_mask)[0]
        
        fig, axes = plt.subplots(2, 10, figsize=(20, 6))
        fig.suptitle('Model Predictions: Top Row = Correct, Bottom Row = Incorrect', fontsize=16)
        
        # Show correct predictions
        for i in range(10):
            if i < len(correct_indices):
                idx = correct_indices[i]
                image = self.X_test[idx].reshape(28, 28)
                axes[0, i].imshow(image, cmap='gray')
                axes[0, i].set_title(f'✅ True: {self.y_test[idx]}\nPred: {predicted_classes[idx]}', 
                                   color='green', fontsize=10)
                axes[0, i].axis('off')
        
        # Show incorrect predictions
        for i in range(10):
            if i < len(incorrect_indices):
                idx = incorrect_indices[i]
                image = self.X_test[idx].reshape(28, 28)
                axes[1, i].imshow(image, cmap='gray')
                axes[1, i].set_title(f'❌ True: {self.y_test[idx]}\nPred: {predicted_classes[idx]}', 
                                   color='red', fontsize=10)
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def predict_single_digit(self, image_data):
        """
        Predict a single digit from image data.
        This is how you'd use the model in a real application!
        """
        if self.model is None:
            print("❌ No trained model available!")
            return None
            
        # Ensure the image is the right shape and normalized
        if image_data.shape != (784,):
            image_data = image_data.reshape(-1)
        
        image_data = image_data.astype('float32') / 255.0
        
        # Make prediction
        prediction = self.model.predict(image_data.reshape(1, -1), verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = prediction[0][predicted_digit]
        
        print(f"🎯 Predicted digit: {predicted_digit}")
        print(f"🔍 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        
        # Show the image
        plt.figure(figsize=(6, 6))
        plt.imshow(image_data.reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {predicted_digit} (Confidence: {confidence:.2f})')
        plt.axis('off')
        plt.show()
        
        return predicted_digit, confidence

# Example usage and demonstration
def main():
    """
    Demonstrate the complete digit recognition pipeline.
    This is like a complete lesson plan from start to finish!
    """
    print("🎉 Welcome to the Advanced Digit Recognition System!")
    print("=" * 60)
    
    # Initialize our recognizer
    recognizer = DigitRecognizer()
    
    # Step 1: Load and explore the data
    print("\n🔄 STEP 1: Loading and exploring data...")
    recognizer.load_data()
    recognizer.visualize_samples()
    
    # Step 2: Build the model
    print("\n🔄 STEP 2: Building the neural network...")
    recognizer.build_model(model_type='cnn')  # Try 'dense' for simpler model
    
    # Step 3: Train the model
    print("\n🔄 STEP 3: Training the model...")
    history = recognizer.train_model(epochs=20, batch_size=128)
    
    # Step 4: Evaluate performance
    print("\n🔄 STEP 4: Evaluating model performance...")
    recognizer.evaluate_model()
    
    # Step 5: Test with a random sample
    print("\n🔄 STEP 5: Testing with a random sample...")
    random_idx = np.random.choice(len(recognizer.X_test))
    test_image = recognizer.X_test[random_idx]
    print(f"True digit: {recognizer.y_test[random_idx]}")
    recognizer.predict_single_digit(test_image)
    
    print("\n🎊 Digit recognition demonstration complete!")
    print("The model is now ready to recognize handwritten digits!")

if __name__ == "__main__":
    main()
