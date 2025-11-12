"""import kagglehub

# Download latest version
path = kagglehub.dataset_download("maysuni/wild-deepfake")

print("Path to dataset files:", path)

"path = /Users/lamiaqamar/.cache/kagglehub/datasets/maysuni/wild-deepfake/versions/1"""

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "/Users/lamiaqamar/.cache/kagglehub/datasets/maysuni/wild-deepfake/versions/1"
IMG_SIZE = (380, 380)
SEQ_LENGTH = 7
BATCH_SIZE = 2


def create_memory_efficient_dataset(split='train', max_samples=10):
    data_dir = os.path.join(DATA_DIR, split)
    print(f"Loading dataset from: {data_dir}")

    def generator():
        real_count = 0
        fake_count = 0
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            label = 0 if 'real' in class_name.lower() else 1
            image_files = [f for f in os.listdir(
                class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for image_file in image_files:
                if (label == 0 and real_count >= max_samples) or (label == 1 and fake_count >= max_samples):
                    continue
                try:
                    image_path = os.path.join(class_path, image_file)
                    image_raw = tf.io.read_file(image_path)
                    image = tf.image.decode_image(image_raw, channels=3)
                    image = tf.image.resize(image, IMG_SIZE)
                    image = tf.cast(image, tf.float32) / 255.0
                    sequence = tf.stack([image] * SEQ_LENGTH, axis=0)
                    if label == 0:
                        real_count += 1
                    else:
                        fake_count += 1
                    yield sequence.numpy(), np.int32(label)
                except Exception:
                    continue

    output_signature = (
        tf.TensorSpec(
            shape=(SEQ_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    return dataset


# Example usage: load small train batch and show images
dataset = create_memory_efficient_dataset('train', max_samples=5)

for x_batch, y_batch in dataset.take(1):
    print("Input batch shape:", x_batch.shape)
    print("Labels batch:", y_batch.numpy())
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    for i in range(x_batch.shape[0]):
        img = x_batch[i, 0].numpy()
        plt.subplot(1, x_batch.shape[0], i + 1)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"Label: {y_batch[i].numpy()}")
        plt.axis('off')
    plt.show()
