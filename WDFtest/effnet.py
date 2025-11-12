import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
DATASET_PATH = "/Users/lamiaqamar/.cache/kagglehub/datasets/maysuni/wild-deepfake/versions/1"
TEST_DIR = os.path.join(DATASET_PATH, "test")
MODEL_PATH = "efficientnetb4_wild_deepfake_best.keras"

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def create_test_generator():
    """Create test data generator for evaluation"""

    # EffNetB4 preprocessing
    def effNet_preprocess(x):
        return tf.keras.applications.efficientnet.preprocess_input(x)

    test_datagen = ImageDataGenerator(
        preprocessing_function=effNet_preprocess
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['real', 'fake'],
        shuffle=False,  # don't shuffle for evaluation
        seed=42
    )

    return test_generator


def load_and_evaluate_model(model_path, test_generator):
    """Load the trained model and get predictions"""

    print("Loading trained EffNetB4 model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

    print("\nModel Architecture Summary:")
    model.summary()

    # Get predictions
    print("\nGenerating predictions on test set...")
    test_generator.reset()

    # Get probabilities
    y_pred_proba = model.predict(test_generator, verbose=1)
    y_pred_proba = y_pred_proba.flatten()

    # Get binary predictions
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Get true labels
    y_true = test_generator.classes[:len(y_pred)]

    print(f"Predictions generated: {len(y_pred)} samples")
    print(
        f"Class distribution - Real: {np.sum(y_true == 0)}, Fake: {np.sum(y_true == 1)}")

    return model, y_true, y_pred, y_pred_proba


def plot_confusion_matrix(y_true, y_pred, save_path="effNetB4_confusion_matrix.png"):
    """Create and plot detailed confusion matrix"""

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot raw counts
    class_names = ['Real', 'Fake']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)\neffNetB4 on WildDeepfake')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    # Plot percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Percentages)\neffNetB4 on WildDeepfake')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed confusion matrix analysis
    print("\n" + "="*60)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*60)

    tn, fp, fn, tp = cm.ravel()

    print(f"True Negatives (Real → Real):  {tn:4d} ({cm_percent[0,0]:.1f}%)")
    print(f"False Positives (Real → Fake): {fp:4d} ({cm_percent[0,1]:.1f}%)")
    print(f"False Negatives (Fake → Real): {fn:4d} ({cm_percent[1,0]:.1f}%)")
    print(f"True Positives (Fake → Fake):  {tp:4d} ({cm_percent[1,1]:.1f}%)")

    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    print(f"\nDerived Metrics:")
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(
        f"Precision:   {precision:.4f} (of predicted fakes, how many were actually fake)")
    print(
        f"Recall:      {recall:.4f} (of actual fakes, how many were detected)")
    print(
        f"Specificity: {specificity:.4f} (of actual reals, how many were correctly identified)")
    print(f"F1-Score:    {f1:.4f}")

    return cm


def plot_roc_curve(y_true, y_pred_proba, save_path="effNetB4_roc_curve.png"):
    """Create and plot ROC curve"""

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Create ROC plot
    plt.figure(figsize=(10, 8))

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.5000)')

    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve - effNetB4 on WildDeepfake Dataset')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Add AUC interpretation
    if roc_auc >= 0.9:
        interpretation = "Excellent"
    elif roc_auc >= 0.8:
        interpretation = "Good"
    elif roc_auc >= 0.7:
        interpretation = "Fair"
    elif roc_auc >= 0.6:
        interpretation = "Poor"
    else:
        interpretation = "Very Poor"

    plt.text(0.6, 0.2, f'AUC: {roc_auc:.4f}\nInterpretation: {interpretation}',
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="lightblue", alpha=0.7),
             fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n" + "="*60)
    print("ROC CURVE ANALYSIS")
    print("="*60)
    print(f"Area Under Curve (AUC): {roc_auc:.4f}")
    print(f"Model Performance: {interpretation}")

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    print(f"\nOptimal Operating Point:")
    print(f"Threshold: {optimal_threshold:.4f}")
    print(f"True Positive Rate: {optimal_tpr:.4f}")
    print(f"False Positive Rate: {optimal_fpr:.4f}")
    print(f"Youden's J Score: {j_scores[optimal_idx]:.4f}")

    return roc_auc, optimal_threshold


def plot_prediction_distribution(y_true, y_pred_proba, save_path="effNetB4_prediction_distribution.png"):
    """Plot distribution of prediction probabilities by class"""

    # Separate probabilities by true class
    real_probs = y_pred_proba[y_true == 0]
    fake_probs = y_pred_proba[y_true == 1]

    plt.figure(figsize=(12, 8))

    # Plot histograms
    plt.hist(real_probs, bins=50, alpha=0.7, color='blue',
             label=f'Real Images (n={len(real_probs)})')
    plt.hist(fake_probs, bins=50, alpha=0.7, color='red',
             label=f'Fake Images (n={len(fake_probs)})')

    # Add decision threshold line
    plt.axvline(x=0.5, color='black', linestyle='--',
                linewidth=2, label='Decision Threshold (0.5)')

    plt.xlabel('Prediction Probability (Fake Score)')
    plt.ylabel('Number of Images')
    plt.title(
        'Distribution of Prediction Probabilities\neffNetB4 on WildDeepfake Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add statistics
    real_mean = np.mean(real_probs)
    fake_mean = np.mean(fake_probs)
    separation = abs(fake_mean - real_mean)

    plt.text(0.02, plt.ylim()[1]*0.8,
             f'Real Images:\nMean: {real_mean:.3f}\nStd: {np.std(real_probs):.3f}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

    plt.text(0.65, plt.ylim()[1]*0.8,
             f'Fake Images:\nMean: {fake_mean:.3f}\nStd: {np.std(fake_probs):.3f}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

    plt.text(0.35, plt.ylim()[1]*0.5,
             f'Class Separation: {separation:.3f}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n" + "="*60)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*60)
    print(
        f"Real Images - Mean: {real_mean:.4f}, Std: {np.std(real_probs):.4f}")
    print(
        f"Fake Images - Mean: {fake_mean:.4f}, Std: {np.std(fake_probs):.4f}")
    print(f"Class Separation: {separation:.4f}")

    return real_probs, fake_probs


def generate_classification_report(y_true, y_pred):
    """Generate detailed classification report"""

    print(f"\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)

    report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'],
                                   output_dict=True)

    # Print formatted report
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake']))

    return report


def main():
    """Main evaluation function"""

    print("="*80)
    print("effNetB4 MODEL EVALUATION ON WILDDEEPFAKE DATASET")
    print("="*80)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please ensure the model file exists in the current directory.")
        return

    # Create test generator
    print("Creating test data generator...")
    test_generator = create_test_generator()
    print(f"Test samples found: {test_generator.samples}")

    # Load model and get predictions
    model, y_true, y_pred, y_pred_proba = load_and_evaluate_model(
        MODEL_PATH, test_generator)

    if model is None:
        return

    # Generate all evaluations
    print("\n" + "="*60)
    print("GENERATING EVALUATION PLOTS AND METRICS")
    print("="*60)

    # 1. Confusion Matrix
    print("\n1. Generating Confusion Matrix...")
    cm = plot_confusion_matrix(y_true, y_pred)

    # 2. ROC Curve
    print("\n2. Generating ROC Curve...")
    roc_auc, optimal_threshold = plot_roc_curve(y_true, y_pred_proba)

    # 3. Prediction Distribution
    print("\n3. Generating Prediction Distribution...")
    real_probs, fake_probs = plot_prediction_distribution(y_true, y_pred_proba)

    # 4. Classification Report
    report = generate_classification_report(y_true, y_pred)

    # Final Summary
    print(f"\n" + "="*80)
    print("FINAL EVALUATION SUMMARY")
    print("="*80)
    print(f"Model: EffNetB4 (WildDeepfake Fine-tuned)")
    print(f"Test Samples: {len(y_true)}")
    print(
        f"Overall Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(
        f"Real Class - Precision: {report['Real']['precision']:.4f}, Recall: {report['Real']['recall']:.4f}")
    print(
        f"Fake Class - Precision: {report['Fake']['precision']:.4f}, Recall: {report['Fake']['recall']:.4f}")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")

    # Save results to file
    with open('EffNetB4_evaluation_results.txt', 'w') as f:
        f.write("EffNetB4 WildDeepfake Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Overall Accuracy: {report['accuracy']:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(
            f"Real Class - Precision: {report['Real']['precision']:.4f}, Recall: {report['Real']['recall']:.4f}\n")
        f.write(
            f"Fake Class - Precision: {report['Fake']['precision']:.4f}, Recall: {report['Fake']['recall']:.4f}\n")
        f.write(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}\n")
        f.write(
            f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}\n")

    print(f"\n📊 All plots saved as PNG files")
    print(f"📝 Results summary saved to: EffNetB4_evaluation_results.txt")
    print("="*80)


if __name__ == "__main__":
    main()
