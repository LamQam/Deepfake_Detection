import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import numpy as np
import random

# Configuration for EfficientNetB7
IMG_SIZE = (600, 600)  # Official input size for EfficientNetB7
BATCH_SIZE = 2  # Reduced due to higher memory requirements
EPOCHS = 20
SEQ_LENGTH = 7


class BalancedVideoGenerator(tf.keras.utils.Sequence):
    def __init__(self, real_dir, fake_dir, batch_size=2):
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

        sequences = []
        labels = []

        for video in batch_real + batch_fake:
            frames = sorted(os.listdir(video))[:SEQ_LENGTH]
            frame_paths = [os.path.join(video, f) for f in frames]
            # EfficientNet preprocessing
            processed_frames = [tf.keras.applications.efficientnet.preprocess_input(
                tf.keras.preprocessing.image.img_to_array(
                    tf.keras.preprocessing.image.load_img(
                        f, target_size=IMG_SIZE)
                )) for f in frame_paths]
            sequences.append(processed_frames)
            labels.append(0 if video in batch_real else 1)

        return np.array(sequences), np.array(labels)


def create_efficientnetb7_model():
    # Create base model
    base_model = EfficientNetB7(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling='avg'
    )
    base_model.trainable = False

    # Build sequence model
    inputs = tf.keras.Input(shape=(SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3))

    # Process each frame
    x = layers.TimeDistributed(base_model)(inputs)

    # Temporal aggregation with regularization
    x = layers.Bidirectional(layers.LSTM(128, recurrent_dropout=0.3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='swish')(
        x)  # Swish works better with EfficientNet
    x = layers.Dropout(0.7)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    # Use lower learning rate for stability
    model.compile(
        optimizer=optimizers.Adam(3e-6),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc', curve='PR'),
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

# Model setup
model = create_efficientnetb7_model()

# Enhanced callbacks
callbacks = [
    ModelCheckpoint('best_effnetb7.keras',
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True,
                    save_weights_only=False),
    EarlyStopping(patience=10,
                  monitor='val_auc',
                  mode='max',
                  restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=3,
        verbose=1,
        mode='max',
        min_lr=1e-7)
]

# Train with mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
