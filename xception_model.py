import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import random

# Configuration
IMG_SIZE = (299, 299)
BATCH_SIZE = 4  # Reduced for memory constraints
EPOCHS = 30
SEQ_LENGTH = 7  # Number of frames per video


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
        # Balance classes explicitly
        batch_real = random.sample(self.real_videos, self.batch_size//2)
        batch_fake = random.sample(self.fake_videos, self.batch_size//2)

        # Load and process frames
        sequences = []
        labels = []

        for video in batch_real + batch_fake:
            frames = sorted(os.listdir(video))[:SEQ_LENGTH]
            frame_paths = [os.path.join(video, f) for f in frames]
            processed_frames = [tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(f, target_size=IMG_SIZE))
                for f in frame_paths]
            sequences.append(processed_frames)
            labels.append(0 if video in batch_real else 1)

        return np.array(sequences), np.array(labels)


def create_xception_model():
    # Create base model
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    # Build sequence model
    inputs = tf.keras.Input(shape=(SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3))

    # Process each frame with Xception
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    # Temporal aggregation
    x = layers.LSTM(128)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='prec'),
            tf.keras.metrics.Recall(name='rec')
        ]
    )
    return model


# Initialize generators
train_gen = BalancedVideoGenerator(
    'deepfake_dataset/LQ/train/real',
    'deepfake_dataset/LQ/train/fake'
)

val_gen = BalancedVideoGenerator(
    'deepfake_dataset/LQ/val/real',
    'deepfake_dataset/LQ/val/fake'
)

# Verify balance
print(
    f"Train videos - Real: {len(train_gen.real_videos)}, Fake: {len(train_gen.fake_videos)}")
print(
    f"Val videos - Real: {len(val_gen.real_videos)}, Fake: {len(val_gen.fake_videos)}")

# Train model
model = create_xception_model()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[
        ModelCheckpoint('best_model.keras', monitor='val_auc',
                        mode='max', save_best_only=True),
        EarlyStopping(patience=5, monitor='val_auc', mode='max')
    ]
)
