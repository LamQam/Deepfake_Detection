# shared_data_utils.py - Updated for streaming
import tensorflow as tf
from datasets import load_dataset
import numpy as np
from PIL import Image

# Global dataset - load once, use everywhere
WILDDEEPFAKE_DS = None


def get_wilddeepfake_dataset():
    """Get streaming dataset"""
    global WILDDEEPFAKE_DS
    if WILDDEEPFAKE_DS is None:
        print("📂 Loading WildDeepfake dataset in streaming mode...")
        WILDDEEPFAKE_DS = load_dataset("xingjunm/WildDeepfake", streaming=True)
        print("✅ Dataset loaded in streaming mode")
    return WILDDEEPFAKE_DS


def create_sequences_generator(split='train', img_size=(380, 380), seq_length=7,
                               preprocessing_fn=None, max_sequences=None):
    """Generic sequence generator for any model"""

    ds = load_dataset("xingjunm/WildDeepfake",
                      split=ds.split, streaming=True)
    # Adjust buffer_size to your RAM and dataset size
    ds = ds.shuffle(buffer_size=3500, seed=42)

    sequence_buffer = []
    label_buffer = []
    last_video_id = None
    sequences_generated = 0

    def extract_video_id(key):
        parts = key.split('/')
        return '/'.join(parts[:4])

    print(
        f"🔄 Creating sequences for {split} (streaming mode, target size: {img_size})...")

    try:
        for example in ds[split]:
            if max_sequences and sequences_generated >= max_sequences:
                break

            try:
                key = example['__key__']
                video_id = extract_video_id(key)
                label = 0 if 'real' in key else 1

                # Apply preprocessing function
                if preprocessing_fn:
                    image = preprocessing_fn(example['png'], img_size)
                else:
                    # Default preprocessing
                    img = example['png'].resize(img_size)
                    image = np.array(img, dtype=np.float32) / 255.0

                if last_video_id is None or video_id == last_video_id:
                    sequence_buffer.append(image)
                    label_buffer.append(label)
                else:
                    if len(sequence_buffer) >= seq_length:
                        yield (np.array(sequence_buffer[:seq_length]), label_buffer[0])
                        sequences_generated += 1

                        # Progress updates
                        if sequences_generated % 100 == 0:
                            print(
                                f"   Generated {sequences_generated} sequences...")

                    sequence_buffer = [image]
                    label_buffer = [label]

                last_video_id = video_id

                # Reduced sliding window to save memory
                if len(sequence_buffer) >= seq_length * 2:  # Only when buffer is large
                    yield (np.array(sequence_buffer[:seq_length]), label_buffer[0])
                    sequences_generated += 1
                    # Half overlap
                    sequence_buffer = sequence_buffer[seq_length//2:]
                    label_buffer = label_buffer[seq_length//2:]

            except Exception as e:
                continue

    except Exception as e:
        print(f"⚠️ Streaming error (this is normal): {e}")

    # Final sequence
    if (sequence_buffer and len(sequence_buffer) >= seq_length and
            (not max_sequences or sequences_generated < max_sequences)):
        yield (np.array(sequence_buffer[:seq_length]), label_buffer[0])
        sequences_generated += 1

    print(f"✅ Generated {sequences_generated} sequences for {split}")

# Preprocessing functions


def efficientnet_preprocessing(image, img_size):
    img = image.resize(img_size)
    img_array = np.array(img, dtype=np.float32)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)


def xception_preprocessing(image, img_size):
    img = image.resize(img_size)
    img_array = np.array(img, dtype=np.float32)
    return tf.keras.applications.xception.preprocess_input(img_array)


def baseline_preprocessing(image, img_size):
    img = image.resize(img_size)
    return np.array(img, dtype=np.float32) / 255.0
