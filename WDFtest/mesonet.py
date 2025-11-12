import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
import glob
import random
from pathlib import Path

# Configuration matching your training setup
IMG_SIZE = (250, 250)  # Match your training config
BATCH_SIZE = 32        # Smaller for testing
DATA_DIR = "/Users/lamiaqamar/.cache/kagglehub/datasets/maysuni/wild-deepfake/versions/1"
TEST_SAMPLES = 1000    # Number of test samples to evaluate

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_test_dataset(num_samples=TEST_SAMPLES):
    """Load and prepare test dataset"""

    print("Loading test dataset...")

    test_dir = os.path.join(DATA_DIR, 'test')

    # Collect test files
    real_pattern = os.path.join(test_dir, 'real', '*.[jpJP]*[gG]')
    fake_pattern = os.path.join(test_dir, 'fake', '*.[jpJP]*[gG]')

    real_files = glob.glob(real_pattern)
    fake_files = glob.glob(fake_pattern)

    print(f"Found {len(real_files)} real and {len(fake_files)} fake test images")

    # Balance and sample
    samples_per_class = min(num_samples // 2, len(real_files), len(fake_files))

    real_files = random.sample(real_files, samples_per_class)
    fake_files = random.sample(fake_files, samples_per_class)

    # Create dataset
    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    # Shuffle
    combined = list(zip(all_files, all_labels))
    random.shuffle(combined)
    all_files, all_labels = zip(*combined)

    print(
        f"Test dataset: {len(all_files)} samples ({samples_per_class} per class)")

    return list(all_files), list(all_labels)


@tf.function
def preprocess_image(image_path):
    """Preprocess image for prediction (matching training preprocessing)"""

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.ensure_shape(image, [None, None, 3])

    # Resize (match training method)
    image = tf.image.resize(image, IMG_SIZE, method='nearest')

    # Normalize
    image = tf.cast(image, tf.float32) / 255.0

    return image


def load_and_predict(model, image_paths, batch_size=BATCH_SIZE):
    """Load images and get predictions in batches"""

    print("Getting model predictions...")

    predictions = []
    total_batches = len(image_paths) // batch_size + \
        (1 if len(image_paths) % batch_size else 0)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]

        # Load and preprocess batch
        batch_images = []
        for path in batch_paths:
            try:
                image = preprocess_image(path)
                batch_images.append(image)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Add dummy image on error
                dummy_image = tf.zeros(
                    [IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.float32)
                batch_images.append(dummy_image)

        if batch_images:
            batch_tensor = tf.stack(batch_images)
            batch_predictions = model.predict(batch_tensor, verbose=0)
            predictions.extend(batch_predictions.flatten())

        # Progress
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed batch {i // batch_size + 1}/{total_batches}")

    return np.array(predictions)


def create_confusion_matrix(y_true, y_pred, threshold=0.5):
    """Create and plot confusion matrix"""

    # Convert predictions to binary
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Add accuracy info
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.figtext(
        0.02, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed metrics
    print("\nConfusion Matrix Analysis:")
    print(f"True Negatives (Real correctly identified): {cm[0, 0]}")
    print(f"False Positives (Real classified as Fake): {cm[0, 1]}")
    print(f"False Negatives (Fake classified as Real): {cm[1, 0]}")
    print(f"True Positives (Fake correctly identified): {cm[1, 1]}")
    print(f"\nAccuracy: {accuracy:.3f}")

    return cm


def create_roc_curve(y_true, y_pred):
    """Create and plot ROC curve"""

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2,
             linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.3f}")

    return roc_auc, optimal_threshold


def create_precision_recall_curve(y_true, y_pred):
    """Create and plot Precision-Recall curve"""

    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)

    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Average Precision Score: {avg_precision:.4f}")

    return avg_precision


def plot_prediction_distribution(y_true, y_pred):
    """Plot distribution of predictions"""

    # Separate predictions by true class
    real_predictions = y_pred[np.array(y_true) == 0]
    fake_predictions = y_pred[np.array(y_true) == 1]

    # Create distribution plot
    plt.figure(figsize=(12, 6))

    # Subplot 1: Histograms
    plt.subplot(1, 2, 1)
    plt.hist(real_predictions, bins=50, alpha=0.7,
             label='Real Images', color='green', density=True)
    plt.hist(fake_predictions, bins=50, alpha=0.7,
             label='Fake Images', color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Prediction Distribution by True Class')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = [real_predictions, fake_predictions]
    box_plot = plt.boxplot(data_to_plot, labels=[
                           'Real', 'Fake'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('green')
    box_plot['boxes'][0].set_alpha(0.7)
    box_plot['boxes'][1].set_facecolor('red')
    box_plot['boxes'][1].set_alpha(0.7)
    plt.ylabel('Prediction Probability')
    plt.title('Prediction Distribution Box Plot')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Statistics
    print(f"\nPrediction Statistics:")
    print(
        f"Real images - Mean: {real_predictions.mean():.3f}, Std: {real_predictions.std():.3f}")
    print(
        f"Fake images - Mean: {fake_predictions.mean():.3f}, Std: {fake_predictions.std():.3f}")

    # Classification at different thresholds
    thresholds = [0.3, 0.5, 0.7]
    print(f"\nClassification results at different thresholds:")
    for thresh in thresholds:
        y_pred_thresh = (y_pred >= thresh).astype(int)
        accuracy = np.mean(y_pred_thresh == y_true)
        precision = np.sum((y_pred_thresh == 1) & (
            y_true == 1)) / np.sum(y_pred_thresh == 1) if np.sum(y_pred_thresh == 1) > 0 else 0
        recall = np.sum((y_pred_thresh == 1) & (y_true == 1)) / \
            np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0
        print(
            f"Threshold {thresh}: Acc={accuracy:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")


def calculate_class_metrics(y_true, y_pred, threshold=0.5):
    """Calculate precision and recall for each class"""

    y_pred_binary = (y_pred >= threshold).astype(int)
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred_binary)

    # Calculate metrics for Real class (class 0)
    real_tp = np.sum((y_true_array == 0) & (
        y_pred_array == 0))  # True positives for real
    real_fp = np.sum((y_true_array == 1) & (
        y_pred_array == 0))  # False positives for real
    real_fn = np.sum((y_true_array == 0) & (
        y_pred_array == 1))  # False negatives for real

    real_precision = real_tp / \
        (real_tp + real_fp) if (real_tp + real_fp) > 0 else 0
    real_recall = real_tp / \
        (real_tp + real_fn) if (real_tp + real_fn) > 0 else 0
    real_f1 = 2 * (real_precision * real_recall) / (real_precision +
                                                    real_recall) if (real_precision + real_recall) > 0 else 0

    # Calculate metrics for Fake class (class 1)
    fake_tp = np.sum((y_true_array == 1) & (
        y_pred_array == 1))  # True positives for fake
    fake_fp = np.sum((y_true_array == 0) & (
        y_pred_array == 1))  # False positives for fake
    fake_fn = np.sum((y_true_array == 1) & (
        y_pred_array == 0))  # False negatives for fake

    fake_precision = fake_tp / \
        (fake_tp + fake_fp) if (fake_tp + fake_fp) > 0 else 0
    fake_recall = fake_tp / \
        (fake_tp + fake_fn) if (fake_tp + fake_fn) > 0 else 0
    fake_f1 = 2 * (fake_precision * fake_recall) / (fake_precision +
                                                    fake_recall) if (fake_precision + fake_recall) > 0 else 0

    return {
        'real': {'precision': real_precision, 'recall': real_recall, 'f1': real_f1, 'support': np.sum(y_true_array == 0)},
        'fake': {'precision': fake_precision, 'recall': fake_recall, 'f1': fake_f1, 'support': np.sum(y_true_array == 1)}
    }


def generate_classification_report(y_true, y_pred, threshold=0.5):
    """Generate detailed classification report with per-class metrics"""

    y_pred_binary = (y_pred >= threshold).astype(int)

    print(f"\nClassification Report (threshold = {threshold}):")
    print("=" * 60)
    report = classification_report(y_true, y_pred_binary,
                                   target_names=['Real', 'Fake'],
                                   digits=3)
    print(report)

    # Calculate and display detailed per-class metrics
    class_metrics = calculate_class_metrics(y_true, y_pred, threshold)

    print(f"\nDETAILED PER-CLASS METRICS:")
    print("=" * 60)
    print(f"REAL CLASS:")
    print(f"  Precision: {class_metrics['real']['precision']:.4f}")
    print(f"  Recall:    {class_metrics['real']['recall']:.4f}")
    print(f"  F1-Score:  {class_metrics['real']['f1']:.4f}")
    print(f"  Support:   {class_metrics['real']['support']}")

    print(f"\nFAKE CLASS:")
    print(f"  Precision: {class_metrics['fake']['precision']:.4f}")
    print(f"  Recall:    {class_metrics['fake']['recall']:.4f}")
    print(f"  F1-Score:  {class_metrics['fake']['f1']:.4f}")
    print(f"  Support:   {class_metrics['fake']['support']}")

    return report, class_metrics


def save_sample_predictions(model, image_paths, y_true, y_pred, num_samples=20):
    """Save sample images with predictions for visual inspection"""

    print(f"\nSaving {num_samples} sample predictions...")

    # Create output directory
    os.makedirs('sample_predictions', exist_ok=True)

    # Get random samples
    indices = random.sample(range(len(image_paths)),
                            min(num_samples, len(image_paths)))

    for i, idx in enumerate(indices):
        try:
            # Load and display image
            image = preprocess_image(image_paths[idx]).numpy()

            # Create plot
            plt.figure(figsize=(6, 4))
            plt.imshow(image)

            # Labels and predictions
            true_label = "FAKE" if y_true[idx] == 1 else "REAL"
            pred_prob = y_pred[idx]
            pred_label = "FAKE" if pred_prob > 0.5 else "REAL"
            confidence = max(pred_prob, 1 - pred_prob)

            # Color code: green for correct, red for incorrect
            color = 'green' if true_label == pred_label else 'red'

            plt.title(f'True: {true_label}, Pred: {pred_label} ({pred_prob:.3f})\n'
                      f'Confidence: {confidence:.3f}', color=color, fontweight='bold')
            plt.axis('off')

            # Save
            filename = f'sample_predictions/sample_{i+1:02d}_{true_label}_{pred_label}_{pred_prob:.3f}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")

    print(f"Sample predictions saved in 'sample_predictions' folder")


def main():
    """Main evaluation function"""

    print("MesoNet Model Evaluation")
    print("=" * 50)

    # Get model path
    model_path = input(
        "Enter model path (default: final_optimized_mesonet.keras): ").strip()
    if not model_path:
        model_path = "final_optimized_mesonet.keras"

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load test dataset
    image_paths, y_true = load_test_dataset(TEST_SAMPLES)

    # Get predictions
    y_pred = load_and_predict(model, image_paths)

    print(f"\nEvaluation Results Summary:")
    print("=" * 50)

    # 1. Confusion Matrix
    print("\n1. Creating Confusion Matrix...")
    confusion_mat = create_confusion_matrix(y_true, y_pred)

    # 2. ROC Curve
    print("\n2. Creating ROC Curve...")
    roc_auc, optimal_threshold = create_roc_curve(y_true, y_pred)

    # 3. Precision-Recall Curve
    print("\n3. Creating Precision-Recall Curve...")
    avg_precision = create_precision_recall_curve(y_true, y_pred)

    # 4. Prediction Distribution
    print("\n4. Plotting Prediction Distribution...")
    plot_prediction_distribution(y_true, y_pred)

    # 5. Classification Report with per-class metrics
    print("\n5. Generating Classification Reports...")
    report_05, class_metrics_05 = generate_classification_report(
        y_true, y_pred, threshold=0.5)
    report_opt, class_metrics_opt = generate_classification_report(
        y_true, y_pred, threshold=optimal_threshold)

    # 6. Save sample predictions
    print("\n6. Saving Sample Predictions...")
    save_sample_predictions(model, image_paths, y_true, y_pred)

    # Final Summary
    print(f"\n" + "=" * 50)
    print("EVALUATION COMPLETE - SUMMARY")
    print("=" * 50)
    print(f"Total samples evaluated: {len(y_true)}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")

    accuracy_05 = np.mean((y_pred >= 0.5) == y_true)
    accuracy_opt = np.mean((y_pred >= optimal_threshold) == y_true)
    print(f"Accuracy @ 0.5 threshold: {accuracy_05:.3f}")
    print(f"Accuracy @ optimal threshold: {accuracy_opt:.3f}")

    # Per-class summary
    print(f"\nPER-CLASS PERFORMANCE @ 0.5 threshold:")
    print(
        f"Real Class - Precision: {class_metrics_05['real']['precision']:.3f}, Recall: {class_metrics_05['real']['recall']:.3f}")
    print(
        f"Fake Class - Precision: {class_metrics_05['fake']['precision']:.3f}, Recall: {class_metrics_05['fake']['recall']:.3f}")

    print(
        f"\nPER-CLASS PERFORMANCE @ optimal threshold ({optimal_threshold:.3f}):")
    print(
        f"Real Class - Precision: {class_metrics_opt['real']['precision']:.3f}, Recall: {class_metrics_opt['real']['recall']:.3f}")
    print(
        f"Fake Class - Precision: {class_metrics_opt['fake']['precision']:.3f}, Recall: {class_metrics_opt['fake']['recall']:.3f}")

    # Performance assessment
    if roc_auc >= 0.90:
        print("\nPerformance: EXCELLENT! 🏆")
    elif roc_auc >= 0.80:
        print("\nPerformance: VERY GOOD! 🥇")
    elif roc_auc >= 0.70:
        print("\nPerformance: GOOD 🥈")
    elif roc_auc >= 0.60:
        print("\nPerformance: MODERATE ⚠️")
    else:
        print("\nPerformance: NEEDS IMPROVEMENT ❌")

    print(f"\nGenerated files:")
    print("- confusion_matrix.png")
    print("- roc_curve.png")
    print("- precision_recall_curve.png")
    print("- prediction_distribution.png")
    print("- sample_predictions/ folder")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    main()
