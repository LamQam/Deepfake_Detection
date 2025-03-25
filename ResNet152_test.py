import tensorflow as tf
import os
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
IMG_SIZE = (224, 224)  # Input size for ResNet-based processing
BATCH_SIZE = 4  # Kept small to manage memory usage
EPOCHS = 30
SEQ_LENGTH = 7  # Number of frames per video sequence


class BalancedVideoGenerator(tf.keras.utils.Sequence):
    """Generates balanced batches of real and fake video sequences for evaluation."""

    def __init__(self, real_dir, fake_dir, batch_size=4):
        self.real_videos = self._get_video_paths(real_dir)
        self.fake_videos = self._get_video_paths(fake_dir)
        self.batch_size = batch_size
        self.indices = list(
            range(max(len(self.real_videos), len(self.fake_videos))))
        random.shuffle(self.indices)  # Randomize batch order

    def _get_video_paths(self, path):
        """Retrieve video directories that contain exactly SEQ_LENGTH frames."""
        return [os.path.join(root, d)
                for root, dirs, _ in os.walk(path)
                for d in dirs if len(os.listdir(os.path.join(root, d))) == SEQ_LENGTH]

    def __len__(self):
        """Return the number of batches per epoch."""
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        """Retrieve a batch of video sequences with corresponding labels."""
        batch_real = random.sample(self.real_videos, self.batch_size // 2)
        batch_fake = random.sample(self.fake_videos, self.batch_size // 2)

        sequences, labels = [], []
        for video in batch_real + batch_fake:
            # Select first SEQ_LENGTH frames
            frames = sorted(os.listdir(video))[:SEQ_LENGTH]
            frame_paths = [os.path.join(video, f) for f in frames]

            # Load and preprocess frames using ResNet152 preprocessing
            processed_frames = [tf.keras.applications.resnet.preprocess_input(
                tf.keras.preprocessing.image.img_to_array(
                    tf.keras.preprocessing.image.load_img(
                        f, target_size=IMG_SIZE)
                )) for f in frame_paths]

            sequences.append(processed_frames)
            # 0 = Real, 1 = Fake
            labels.append(0 if video in batch_real else 1)

        return np.array(sequences), np.array(labels)


# Initialize test set generator
test_gen = BalancedVideoGenerator(
    'deepfake_dataset/LQ/test/real',
    'deepfake_dataset/LQ/test/fake'
)

# Print dataset statistics
print(
    f"\nTest videos - Real: {len(test_gen.real_videos)}, Fake: {len(test_gen.fake_videos)}")


# Load the best trained model
best_model = tf.keras.models.load_model('best_model_resnet.keras')

# Evaluate model on test set
print("\nEvaluating on test set...")
test_results = best_model.evaluate(test_gen)
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test AUC: {test_results[1]:.4f}")
print(f"Test Precision: {test_results[2]:.4f}")
print(f"Test Recall: {test_results[3]:.4f}")

# Generate predictions for further analysis
y_true, y_pred_probs = [], []

for i in range(len(test_gen)):
    X, y = test_gen[i]
    y_true.extend(y)
    y_pred_probs.extend(best_model.predict(X, verbose=0).flatten())

y_pred = (np.array(y_pred_probs) > 0.5).astype(
    int)  # Convert probabilities to binary labels

# Classification Report
class_names = ['Real', 'Fake']
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix visualization
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Test Set Confusion Matrix')
plt.show()

# ROC Curve visualization
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2,
         linestyle='--')  # Reference diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
