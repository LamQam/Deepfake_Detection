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
    def __init__(self, real_dir, fake_dir, batch_size=4):
        self.real_videos = self._get_video_paths(real_dir)
        self.fake_videos = self._get_video_paths(fake_dir)
        self.batch_size = batch_size
        self.indices = list(
            range(max(len(self.real_videos), len(self.fake_videos))))
        random.shuffle(self.indices)

    def _get_video_paths(self, path):
        return [os.path.join(root, d)
                for root, dirs, _ in os.walk(path)
                for d in dirs if len(os.listdir(os.path.join(root, d))) == SEQ_LENGTH]

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_real = random.sample(self.real_videos, self.batch_size//2)
        batch_fake = random.sample(self.fake_videos, self.batch_size//2)
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

                # Data augmentation
                if np.random.rand() > 0.5:
                    img = tf.image.flip_left_right(img)
                img = tf.image.random_brightness(img, max_delta=0.1)
                img = tf.image.random_contrast(img, 0.8, 1.2)
                img = tf.image.random_saturation(img, 0.8, 1.2)

                processed_frames.append(img / 255.0)

            sequences.append(processed_frames)
            labels.append(0 if video in real_paths else 1)

        return np.array(sequences), np.array(labels)


def _meso_inception_block():
    input_img = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # First Inception
    x1 = layers.Conv2D(1, (1, 1), padding='same')(input_img)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    x2 = layers.Conv2D(1, (1, 1), padding='same')(input_img)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(3, (3, 3), padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    x3 = layers.Conv2D(1, (1, 1), padding='same')(input_img)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(3, (5, 5), padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)

    x4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(
        1, 1), padding='same')(input_img)
    x4 = layers.Conv2D(1, (1, 1), padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)

    # Second Inception
    x1 = layers.Conv2D(1, (1, 1), padding='same')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    x2 = layers.Conv2D(1, (1, 1), padding='same')(x)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(3, (3, 3), padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    x3 = layers.Conv2D(1, (1, 1), padding='same')(x)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(3, (5, 5), padding='same')(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)

    x4 = layers.MaxPooling2D(pool_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x4 = layers.Conv2D(1, (1, 1), padding='same')(x4)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)

    x = layers.concatenate([x1, x2, x3, x4], axis=-1)

    # Additional layers
    x = layers.Conv2D(16, (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(16, (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    return models.Model(input_img, x)


def evaluate_mesonet():
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
    print(f"\nTest Loss: {test_results[0]:.4f}")
    print(f"Test AUC: {test_results[1]:.4f}")
    print(f"Test Precision: {test_results[2]:.4f}")
    print(f"Test Recall: {test_results[3]:.4f}")

    # Generate predictions
    y_true, y_pred_probs = [], []
    for i in range(len(test_gen)):
        X, y = test_gen[i]
        y_true.extend(y)
        y_pred_probs.extend(model.predict(X, verbose=0).flatten())

    y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

    # Classification Report
    class_names = ['Real', 'Fake']
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test Set Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    evaluate_mesonet()
