import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
import random

# Configuration
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 50
SEQ_LENGTH = 7
LEARNING_RATE = 1e-4
MIXUP_ALPHA = 0.2  # Mixup regularization
LABEL_SMOOTHING = 0.1  # Helps prevent overconfidence in predictions


def get_video_paths(root_dir):
    """
    Traverse the directory and return paths of video folders 
    containing exactly SEQ_LENGTH frames.
    """
    video_paths = []
    for root, dirs, files in os.walk(root_dir):
        if len(dirs) == 0 and len(os.listdir(root)) == SEQ_LENGTH:
            video_paths.append(root)
    return video_paths


class BalancedVideoGenerator(tf.keras.utils.Sequence):
    def __init__(self, real_dir, fake_dir, batch_size=BATCH_SIZE, augment=True, mixup_alpha=MIXUP_ALPHA):
        self.real_videos = get_video_paths(real_dir)
        self.fake_videos = get_video_paths(fake_dir)
        self.batch_size = batch_size
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.indices = list(
            range(max(len(self.real_videos), len(self.fake_videos))))
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        # Sample balanced batches from real and fake videos
        batch_real = random.sample(self.real_videos, self.batch_size // 2)
        batch_fake = random.sample(self.fake_videos, self.batch_size // 2)
        real_set = set(batch_real)

        sequences, labels = [], []
        for video in batch_real + batch_fake:
            # Load the first SEQ_LENGTH frames
            frames = sorted(os.listdir(video))[:SEQ_LENGTH]
            frame_paths = [os.path.join(video, f) for f in frames]
            processed_frames = []

            for f in frame_paths:
                img = tf.keras.preprocessing.image.load_img(
                    f, target_size=IMG_SIZE)
                img = tf.keras.preprocessing.image.img_to_array(img)

                if self.augment:
                    # Apply data augmentation
                    if np.random.rand() > 0.5:
                        img = tf.image.flip_left_right(img)
                    img = tf.image.random_brightness(img, max_delta=0.1)
                    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
                    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
                    if np.random.rand() > 0.7:
                        img = tf.image.rot90(img, k=random.randint(1, 3))
                processed_frames.append(img / 255.0)

            sequences.append(processed_frames)
            labels.append(0 if video in real_set else 1)

        X = np.array(sequences)
        y = np.array(labels).astype(np.float32)

        # Apply mixup augmentation if enabled
        if self.augment and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = np.random.permutation(X.shape[0])
            X = lam * X + (1 - lam) * X[index]
            y = lam * y + (1 - lam) * y[index]

        return X, y

# MesoInception-based feature extractor


def _meso_inception_block():
    input_img = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Two Inception-like branches with different kernel sizes
    x1 = layers.Conv2D(1, (1, 1), padding='same')(input_img)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    x2 = layers.Conv2D(1, (1, 1), padding='same')(input_img)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(3, (3, 3), padding='same')(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    x = layers.concatenate([x1, x2], axis=-1)
    x = layers.Conv2D(16, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)

    return models.Model(input_img, x)

# TimeDistributed MesoInception model


def MesoInception4():
    input_layer = layers.Input(shape=(SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.TimeDistributed(_meso_inception_block())(input_layer)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(input_layer, outputs)


# Create train, val, test generators
train_gen = BalancedVideoGenerator(
    'deepfake_dataset/LQ/train/real', 'deepfake_dataset/LQ/train/fake', augment=True)
val_gen = BalancedVideoGenerator(
    'deepfake_dataset/LQ/val/real', 'deepfake_dataset/LQ/val/fake', augment=False)

# Compute class weights to handle imbalance
class_weights = {
    0: len(train_gen.real_videos) / (len(train_gen.real_videos) + len(train_gen.fake_videos)),
    1: len(train_gen.fake_videos) / (len(train_gen.real_videos) + len(train_gen.fake_videos))
}

# Compile model
model = MesoInception4()
model.compile(
    optimizer=optimizers.Adam(LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(
        name='prec'), tf.keras.metrics.Recall(name='rec')]
)

# Callbacks for training efficiency
callbacks = [
    ModelCheckpoint('best_meso_inception.keras',
                    monitor='val_auc', mode='max', save_best_only=True),
    EarlyStopping(patience=7, monitor='val_auc', mode='max'),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6)
]

# Train the model
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=EPOCHS, class_weight=class_weights, callbacks=callbacks)
