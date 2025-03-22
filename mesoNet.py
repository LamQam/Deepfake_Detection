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
MIXUP_ALPHA = 0.2  # Set to 0 to disable mixup
LABEL_SMOOTHING = 0.1


def get_video_paths(root_dir):
    """
    Traverse the directory tree and return all video directories 
    that contain exactly SEQ_LENGTH frames.
    Expected structure:
        root_dir/subjectX/videoY/frame*.jpg
    """
    video_paths = []
    for root, dirs, files in os.walk(root_dir):
        # We expect the video folder to contain exactly SEQ_LENGTH frames.
        # Skip directories that are subjects (they typically contain subdirectories).
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
        # Create a combined index list based on the maximum number of videos per class.
        self.indices = list(
            range(max(len(self.real_videos), len(self.fake_videos))))
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        # Sample balanced mini-batches from real and fake videos
        batch_real = random.sample(self.real_videos, self.batch_size // 2)
        batch_fake = random.sample(self.fake_videos, self.batch_size // 2)
        real_set = set(batch_real)

        sequences, labels = [], []
        for video in batch_real + batch_fake:
            # Load the first SEQ_LENGTH frames (assuming sorted order)
            frames = sorted(os.listdir(video))[:SEQ_LENGTH]
            frame_paths = [os.path.join(video, f) for f in frames]
            processed_frames = []
            for f in frame_paths:
                img = tf.keras.preprocessing.image.load_img(
                    f, target_size=IMG_SIZE)
                img = tf.keras.preprocessing.image.img_to_array(img)
                if self.augment:
                    # Data augmentation: flip, brightness, contrast, saturation
                    if np.random.rand() > 0.5:
                        img = tf.image.flip_left_right(img)
                    img = tf.image.random_brightness(img, max_delta=0.1)
                    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
                    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
                    # Additional augmentation: random rotation by 90° increments
                    if np.random.rand() > 0.7:
                        # rotate 90, 180, or 270 degrees
                        k = random.randint(1, 3)
                        img = tf.image.rot90(img, k=k)
                processed_frames.append(img / 255.0)

            sequences.append(processed_frames)
            # Label: 0 for real, 1 for fake
            labels.append(0 if video in real_set else 1)

        # Convert to numpy arrays
        X = np.array(sequences)  # Shape: (batch_size, SEQ_LENGTH, 256, 256, 3)
        y = np.array(labels).astype(np.float32)  # Convert to float for mixup

        # Apply mixup augmentation if enabled and if augment is True
        if self.augment and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = np.random.permutation(X.shape[0])
            X = lam * X + (1 - lam) * X[index]
            y = lam * y + (1 - lam) * y[index]

        return X, y

# Define the MesoInception block (adapted from the repository)


def _meso_inception_block():
    input_img = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # First Inception branch
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

    # Second Inception branch
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

    # Additional convolutional layers with L2 regularization
    x = layers.Conv2D(16, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

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


def MesoInception4():
    # The input is a sequence of SEQ_LENGTH frames
    input_layer = layers.Input(shape=(SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3))
    # Apply the same feature extractor on each frame
    x = layers.TimeDistributed(_meso_inception_block())(input_layer)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(input_layer, outputs)


# Create generators for Train, Val, and Test sets.
# For training we enable augmentation (and mixup) while for validation and testing we disable it.
train_gen = BalancedVideoGenerator(
    'deepfake_dataset/LQ/train/real',
    'deepfake_dataset/LQ/train/fake',
    augment=True,
    mixup_alpha=MIXUP_ALPHA
)
val_gen = BalancedVideoGenerator(
    'deepfake_dataset/LQ/val/real',
    'deepfake_dataset/LQ/val/fake',
    augment=False  # disable augmentation for validation
)
test_gen = BalancedVideoGenerator(
    'deepfake_dataset/LQ/test/real',
    'deepfake_dataset/LQ/test/fake',
    augment=False  # disable augmentation for testing
)


def verify_generator(gen, name="Generator"):
    X, y = gen[0]
    print(f"{name} batch shape: {X.shape}, Label distribution: {np.unique(y, return_counts=True)}")


print("Training generator sample:")
verify_generator(train_gen, "Train")
print("\nValidation generator sample:")
verify_generator(val_gen, "Validation")
print("\nTest generator sample:")
verify_generator(test_gen, "Test")

# Compute class weights for training balance
total_train = len(train_gen.real_videos) + len(train_gen.fake_videos)
class_weights = {
    0: total_train / (2 * len(train_gen.real_videos)),
    1: total_train / (2 * len(train_gen.fake_videos))
}

# Build and compile the model.
model = MesoInception4()
model.compile(
    optimizer=optimizers.Adam(LEARNING_RATE),
    # Use BinaryCrossentropy with label smoothing to regularize predictions.
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='prec'),
        tf.keras.metrics.Recall(name='rec')
    ]
)

# Set up callbacks for checkpointing, early stopping, and reducing learning rate on plateau.
callbacks = [
    ModelCheckpoint('best_meso_inception.keras',
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True),
    EarlyStopping(patience=7, monitor='val_auc', mode='max'),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6)
]

# Train the model.
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# After training, evaluate on the test set.
test_loss, test_auc, test_prec, test_rec = model.evaluate(test_gen)
print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f} | Test Precision: {test_prec:.4f} | Test Recall: {test_rec:.4f}")
