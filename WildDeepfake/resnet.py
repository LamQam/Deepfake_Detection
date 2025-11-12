import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Research Configuration for ResNet152
IMG_SIZE = (224, 224)  # ResNet152's standard input size
BATCH_SIZE = 16  # Optimized for ResNet152 and MacBook memory
EPOCHS = 20  # Reasonable for research
LEARNING_RATE = 2e-4  # Higher learning rate for better convergence
FINE_TUNE_EPOCHS = 10
USE_SUBSET = True  # Set to False for full dataset
SUBSET_SIZE = 50000  # Balanced subset for faster experimentation

# Dataset paths
DATASET_PATH = "/Users/lamiaqamar/.cache/kagglehub/datasets/maysuni/wild-deepfake/versions/1"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "valid")
TEST_DIR = os.path.join(DATASET_PATH, "test")


def create_resnet152_data_generators():
    """Create data generators optimized for ResNet152 research."""

    # ResNet152 preprocessing function
    def resnet_preprocess(x):
        """Preprocess input for ResNet152."""
        return tf.keras.applications.resnet.preprocess_input(x)

    # Training augmentation - research-appropriate for deepfakes
    train_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess,
        rotation_range=15,  # Moderate rotation for face images
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],  # Brightness variation
        fill_mode='nearest'
    )

    # Validation and test - no augmentation
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=resnet_preprocess
    )

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake'],
        shuffle=True,
        seed=42
    )

    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake'],
        shuffle=False,
        seed=42
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake'],
        shuffle=False,
        seed=42
    )

    return train_generator, val_generator, test_generator


def extract_resnet152_from_video_model(video_model_path):
    """Extract ResNet152 feature extractor from your trained video model."""
    try:
        # Load the trained video model
        video_model = tf.keras.models.load_model(video_model_path)

        # Extract the ResNet152 base model (it's inside TimeDistributed layer)
        time_distributed_layer = None
        for layer in video_model.layers:
            if isinstance(layer, tf.keras.layers.TimeDistributed):
                time_distributed_layer = layer
                break

        if time_distributed_layer is not None:
            # Get the ResNet152 model from TimeDistributed
            resnet152_model = time_distributed_layer.layer
            print("Successfully extracted ResNet152 from video model")
            return resnet152_model
        else:
            print("Could not find TimeDistributed layer, creating new ResNet152")
            return None

    except Exception as e:
        print(f"Error loading video model: {e}")
        print("Creating new ResNet152 model...")
        return None


def create_resnet152_research_model(pretrained_model_path=None):
    """Create ResNet152 model optimized for research on deepfakes."""

    # Try to extract feature extractor from your video model
    base_model = None
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        base_model = extract_resnet152_from_video_model(pretrained_model_path)

    # If extraction failed, create new ResNet152
    if base_model is None:
        base_model = ResNet152(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling='avg'  # Global Average Pooling
        )
        print(
            f"Using fresh ResNet152 with ImageNet weights, input size: {IMG_SIZE}")

    # Freeze base model initially
    base_model.trainable = False

    # Research-appropriate classification head
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # Feature extraction
    x = base_model(inputs, training=False)

    # Classification layers - optimized for deepfake detection
    x = layers.Dense(1024, activation='relu')(
        x)  # Larger first layer for ResNet152
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate=LEARNING_RATE):
    """Compile model with research-appropriate metrics."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model


def get_training_steps(train_gen, val_gen):
    """Calculate training steps for subset or full dataset."""
    if USE_SUBSET:
        steps_per_epoch = min(SUBSET_SIZE // BATCH_SIZE, len(train_gen))
        # 1/5 of training steps
        validation_steps = min(len(val_gen), steps_per_epoch // 5)
        print(f"Using subset training:")
        print(f"  - Training steps per epoch: {steps_per_epoch}")
        print(f"  - Validation steps: {validation_steps}")
        print(
            f"  - Approximate samples per epoch: {steps_per_epoch * BATCH_SIZE}")
    else:
        steps_per_epoch = len(train_gen)
        validation_steps = len(val_gen)
        print(f"Using full dataset:")
        print(f"  - Training steps per epoch: {steps_per_epoch}")
        print(f"  - Validation steps: {validation_steps}")

    return steps_per_epoch, validation_steps


def plot_resnet152_history(history, fine_tune_history=None, save_path='resnet152_research_history.png'):
    """Plot comprehensive training history for ResNet152 research documentation."""

    # Combine histories if fine-tuning was performed
    if fine_tune_history is not None:
        combined_history = {}
        for key in history.history.keys():
            combined_history[key] = history.history[key] + \
                fine_tune_history.history[key]

        # Mark where fine-tuning started
        finetune_start = len(history.history['loss'])
    else:
        combined_history = history.history
        finetune_start = None

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot accuracy
    axes[0, 0].plot(combined_history['accuracy'],
                    'b-', label='Training Accuracy')
    axes[0, 0].plot(combined_history['val_accuracy'],
                    'r-', label='Validation Accuracy')
    if finetune_start:
        axes[0, 0].axvline(x=finetune_start, color='green',
                           linestyle='--', alpha=0.7, label='Fine-tuning starts')
    axes[0, 0].set_title('ResNet152 - Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot loss
    axes[0, 1].plot(combined_history['loss'], 'b-', label='Training Loss')
    axes[0, 1].plot(combined_history['val_loss'],
                    'r-', label='Validation Loss')
    if finetune_start:
        axes[0, 1].axvline(x=finetune_start, color='green',
                           linestyle='--', alpha=0.7, label='Fine-tuning starts')
    axes[0, 1].set_title('ResNet152 - Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot AUC
    axes[1, 0].plot(combined_history['auc'], 'b-', label='Training AUC')
    axes[1, 0].plot(combined_history['val_auc'], 'r-', label='Validation AUC')
    if finetune_start:
        axes[1, 0].axvline(x=finetune_start, color='green',
                           linestyle='--', alpha=0.7, label='Fine-tuning starts')
    axes[1, 0].set_title('ResNet152 - AUC Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot Precision and Recall
    axes[1, 1].plot(combined_history['precision'],
                    'b-', label='Training Precision')
    axes[1, 1].plot(combined_history['val_precision'],
                    'r-', label='Validation Precision')
    axes[1, 1].plot(combined_history['recall'], 'g-', label='Training Recall')
    axes[1, 1].plot(combined_history['val_recall'],
                    'orange', label='Validation Recall')
    if finetune_start:
        axes[1, 1].axvline(x=finetune_start, color='green',
                           linestyle='--', alpha=0.7, label='Fine-tuning starts')
    axes[1, 1].set_title('ResNet152 - Precision & Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_resnet152_model(model, test_generator):
    """Comprehensive evaluation for ResNet152 research documentation."""
    print("="*60)
    print("EVALUATING RESNET152 ON WILD DEEPFAKE DATASET")
    print("="*60)

    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = test_generator.classes[:len(predicted_classes)]

    # Calculate metrics
    test_loss, test_acc, test_auc, test_prec, test_rec = model.evaluate(
        test_generator, verbose=0)
    f1_score = 2 * (test_prec * test_rec) / (test_prec +
                                             test_rec) if (test_prec + test_rec) > 0 else 0

    print(f"\nRESNET152 PERFORMANCE METRICS:")
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test AUC:       {test_auc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall:    {test_rec:.4f}")
    print(f"Test F1-Score:  {f1_score:.4f}")
    print(f"Test Loss:      {test_loss:.4f}")

    # Detailed classification report
    class_names = ['Real', 'Fake']
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(true_classes,
          predicted_classes, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('ResNet152 - Confusion Matrix\nWild Deepfake Dataset')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('resnet152_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'accuracy': test_acc,
        'auc': test_auc,
        'precision': test_prec,
        'recall': test_rec,
        'f1_score': f1_score,
        'loss': test_loss
    }


def main():
    """Research-focused training pipeline for ResNet152."""
    print("🔬 RESEARCH: ResNet152 Transfer Learning on Wild Deepfake Dataset")
    print(f"Configuration: {IMG_SIZE} input, batch size {BATCH_SIZE}")
    print(f"Subset mode: {'ON' if USE_SUBSET else 'OFF'}")
    print("="*70)

    # Create data generators
    print("Loading ResNet152 data generators...")
    "train_gen, val_gen,"
    test_gen = create_resnet152_data_generators()

    print(f"Dataset loaded:")
    # print(f"  Training samples: {train_gen.samples}")
    # print(f"  Validation samples: {val_gen.samples}")
    print(f"  Test samples: {test_gen.samples}")

    # Calculate training steps
    # steps_per_epoch, validation_steps = get_training_steps(train_gen, val_gen)

    # Create ResNet152 model
    print("\nCreating ResNet152 model...")
    model = create_resnet152_research_model(
        'resnet152_wild_deepfake_best.keras')  # Your video model path
    model = compile_model(model)

    print("\nResNet152 Model Architecture:")
    model.summary()

    # Callbacks for research training
"""     callbacks = [
        ModelCheckpoint(
            'resnet152_wild_deepfake_best.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            patience=8,
            monitor='val_auc',
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1
        )
    ] """

# Phase 1: Initial training with frozen base

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Phase 2: Fine-tuning
""" print("\n" + "="*50)
        print("PHASE 2: FINE-TUNING RESNET152")
        print("="*50)

        # Unfreeze top layers of ResNet152
        base_model.trainable = True

        # ResNet152 has many layers, unfreeze last 50 layers
        fine_tune_at = len(base_model.layers) - 50

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        print(f"Fine-tuning last {len(base_model.layers) - fine_tune_at} layers")

        # Recompile with lower learning rate
        model = compile_model(model, learning_rate=LEARNING_RATE/10)

        fine_tune_callbacks = [
            ModelCheckpoint(
                'resnet152_wild_deepfake_finetuned.keras',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                patience=6,
                monitor='val_auc',
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-8,
                verbose=1
            )
        ]

        fine_tune_history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=fine_tune_callbacks,
            verbose=1
        )

        # Plot training history
        print("\nGenerating ResNet152 training history plots...")
        plot_resnet152_history(history, fine_tune_history)
     """
# Final evaluation
results = evaluate_resnet152_model(model, test_gen)

# Save final model
model.save('resnet152_wild_deepfake_final.keras')

# Research summary
print("\n" + "="*70)
print("RESEARCH SUMMARY - RESNET152 ON WILD DEEPFAKE DATASET")
print("="*70)
print(f"Model: ResNet152")
print(f"Dataset: Wild Deepfake (Kaggle)")
print(f"Input Size: {IMG_SIZE}")
print(f"Training Strategy: Transfer Learning + Fine-tuning")
print(
    f"Final Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
print(f"Final Test AUC: {results['auc']:.4f}")
print(f"Final Test F1-Score: {results['f1_score']:.4f}")
print("="*70)
print("✅ ResNet152 research experiment completed successfully!")


if __name__ == "__main__":
    # Set memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    main()
