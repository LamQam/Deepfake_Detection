import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# Configuration
IMG_SIZE = (256, 256)
SEQ_LENGTH = 7
BATCH_SIZE = 4


class BalancedVideoGenerator(tf.keras.utils.Sequence):
    """Generates balanced batches of real and fake videos for training/testing."""

    def __init__(self, real_dir, fake_dir, batch_size=4):
        self.real_videos = self._get_video_paths(real_dir)
        self.fake_videos = self._get_video_paths(fake_dir)
        self.batch_size = batch_size
        self.indices = list(
            range(max(len(self.real_videos), len(self.fake_videos))))
        random.shuffle(self.indices)

    def _get_video_paths(self, path):
        """Retrieves video directories containing exactly SEQ_LENGTH frames."""
        return [os.path.join(root, d)
                for root, dirs, _ in os.walk(path)
                for d in dirs if len(os.listdir(os.path.join(root, d))) == SEQ_LENGTH]

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        """Loads a balanced batch of real and fake videos."""
        batch_real = random.sample(self.real_videos, self.batch_size // 2)
        batch_fake = random.sample(self.fake_videos, self.batch_size // 2)
        real_paths = set(batch_real)

        sequences, labels = [], []
        for video in batch_real + batch_fake:
            frames = sorted(os.listdir(video))[:SEQ_LENGTH]
            frame_paths = [os.path.join(video, f) for f in frames]

            processed_frames = []
            for f in frame_paths:
                img = tf.keras.preprocessing.image.load_img(
                    f, target_size=IMG_SIZE)
                img = tf.keras.preprocessing.image.img_to_array(img)

                # Apply random data augmentations
                if np.random.rand() > 0.5:
                    img = tf.image.flip_left_right(img)
                img = tf.image.random_brightness(img, max_delta=0.1)
                img = tf.image.random_contrast(img, 0.8, 1.2)
                img = tf.image.random_saturation(img, 0.8, 1.2)

                processed_frames.append(img / 255.0)

            sequences.append(processed_frames)
            labels.append(0 if video in real_paths else 1)

        return np.array(sequences), np.array(labels)


def evaluate_mesonet():
    """Loads the trained MesoNet model and evaluates it on the test dataset."""
    test_gen = BalancedVideoGenerator(
        'deepfake_dataset/LQ/test/real',
        'deepfake_dataset/LQ/test/fake',
        batch_size=BATCH_SIZE
    )

    print(
        f"Test videos - Real: {len(test_gen.real_videos)}, Fake: {len(test_gen.fake_videos)}")

    model = tf.keras.models.load_model('weights/best_meso_inception.keras')

    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_gen)
    print(
        f"\nTest Loss: {test_results[0]:.4f}, Test AUC: {test_results[1]:.4f}, Precision: {test_results[2]:.4f}, Recall: {test_results[3]:.4f}")

    # Generate predictions
    y_true, y_pred_probs = [], []
    for i in range(len(test_gen)):
        X, y = test_gen[i]
        y_true.extend(y)
        y_pred_probs.extend(model.predict(X, verbose=0).flatten())

    y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[
                'Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_mesonet()
