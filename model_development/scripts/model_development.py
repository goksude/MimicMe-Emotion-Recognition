import matplotlib # Import matplotlib first
matplotlib.use('Agg') # Set the backend BEFORE importing pyplot or seaborn

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight # For calculating class weights
import seaborn as sns
from math import ceil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DEV_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(MODEL_DEV_ROOT)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data_prepared_for_model') # dataset_prepare.py'nin oluşturduğu klasör
TRAIN_DIR = os.path.join(DATA_ROOT, 'train')
VAL_DIR = os.path.join(DATA_ROOT, 'test')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'saved_model', 'emotion_model.keras')
PLOT_SAVE_PATH = os.path.join(PROJECT_ROOT, 'saved_model', 'training_plot.png')
CONFUSION_MATRIX_PLOT_PATH = os.path.join(PROJECT_ROOT, 'saved_model', 'confusion_matrix.png')

IMAGE_HEIGHT, IMAGE_WIDTH = 48, 48
BATCH_SIZE = 64
NUM_EPOCHS = 200

# --- Helper Function: Plot Model History ---
def plot_model_history(model_history, save_plot_path=PLOT_SAVE_PATH):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    if 'accuracy' in model_history.history and 'val_accuracy' in model_history.history:
        axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'], label='Train Accuracy')
        axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'], label='Validation Accuracy')
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axs[0].legend(loc='best')
    else:
        axs[0].set_title('Accuracy data not available')
        print("Warning: 'accuracy' or 'val_accuracy' not found in model history.")

    if 'loss' in model_history.history and 'val_loss' in model_history.history:
        axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'], label='Train Loss')
        axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'], label='Validation Loss')
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        axs[1].legend(loc='best')
    else:
        axs[1].set_title('Loss data not available')
        print("Warning: 'loss' or 'val_loss' not found in model history.")
    try:
        fig.savefig(save_plot_path)
        print(f"Training history plot saved to {os.path.abspath(save_plot_path)}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig)

# --- Helper Function: Plot Confusion Matrix ---
def plot_confusion_matrix(cm, class_names, save_path=CONFUSION_MATRIX_PLOT_PATH):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    try:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")
    plt.close()

# --- Main Training and Evaluation Function ---
def train_and_evaluate_emotion_model():
    print("Starting emotion model training (with CLIPPED class weights) and evaluation...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"ERROR: Training ({TRAIN_DIR}) or Validation ({VAL_DIR}) directory not found.")
        return

    train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical',
            shuffle=True)

    validation_generator = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical',
            shuffle=False)

    class_indices = train_generator.class_indices
    print(f"\nTraining classes (from generator): {class_indices}")
    class_names = [name for name, index in sorted(class_indices.items(), key=lambda item: item[1])]
    print(f"Ordered class names for evaluation: {class_names}")
    num_classes = len(class_names)

    # --- Calculate Class Weights AND CLIP THEM ---
    class_weights_dict = None # Initialize
    if train_generator.samples > 0 :
        unique_classes, class_counts_from_generator = np.unique(train_generator.classes, return_counts=True)
        print("\n--- Class Distribution in Training Set (from generator) ---")
        temp_class_map = {v: k for k, v in class_indices.items()}
        for i, class_idx in enumerate(unique_classes):
            print(f"  Class {class_idx} ({temp_class_map.get(class_idx, 'Unknown')}): {class_counts_from_generator[i]} samples")

        class_weights_calculated = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_generator.classes
        )
        raw_class_weights_dict = dict(zip(unique_classes, class_weights_calculated))
        print("\nRaw Calculated Class Weights:")
        for class_idx, weight in raw_class_weights_dict.items():
            print(f"  Class {class_idx} ({temp_class_map.get(class_idx, 'Unknown')}): {weight:.4f}")

        # --- CLIPPING LOGIC ---
        MAX_WEIGHT = 4.0
        class_weights_dict = {
            class_idx: min(weight, MAX_WEIGHT)
            for class_idx, weight in raw_class_weights_dict.items()
        }
        print("\nClipped Class Weights (Max Weight = {}):".format(MAX_WEIGHT))
        for class_idx, weight in class_weights_dict.items():
            print(f"  Class {class_idx} ({temp_class_map.get(class_idx, 'Unknown')}): {weight:.4f}")
        # --- END OF CLIPPING LOGIC ---

    else:
        print("Warning: No training samples found by generator. Cannot calculate class weights. Training without explicit class weights.")


    # --- Create the model ---
    model = Sequential([
        Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    print("\n--- Model Summary ---")
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-7)
    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    num_train_actual = train_generator.samples
    num_val_actual = validation_generator.samples
    print(f"\nActual number of training samples: {num_train_actual}")
    print(f"Actual number of validation samples: {num_val_actual}")

    if num_train_actual == 0:
        print("ERROR: No training images found.")
        return
    if num_val_actual == 0:
        print("ERROR: No validation images found. Cannot proceed with validation and testing.")
        return

    print("\n--- Starting Training ---")
    fit_params = {
        'callbacks': callbacks_list
    }
    if class_weights_dict: # Only add class_weight if it was successfully calculated
        fit_params['class_weight'] = class_weights_dict
        print("Training with CLIPPED class weights.")
    else:
        print("Training without explicit class weights (either no samples or issue calculating weights).")


    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train_actual // BATCH_SIZE if BATCH_SIZE > 0 else 1,
            epochs=NUM_EPOCHS,
            validation_data=validation_generator,
            validation_steps=num_val_actual // BATCH_SIZE if BATCH_SIZE > 0 else 1,
            **fit_params
            )

    print("\n--- Training Finished ---")
    plot_model_history(model_info)
    loaded_model = model

    print("\n--- Evaluating Model on Test/Validation Set ---")
    loss, accuracy = loaded_model.evaluate(validation_generator, steps=ceil(num_val_actual / BATCH_SIZE))
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    print("\nGenerating predictions for detailed evaluation...")
    validation_generator.reset()
    predictions_probs = loaded_model.predict(validation_generator, steps=ceil(num_val_actual / BATCH_SIZE))
    y_pred = np.argmax(predictions_probs[:num_val_actual], axis=1)
    y_true = validation_generator.classes

    if len(y_true) != len(y_pred):
        print(f"Warning: Mismatch in length of true labels ({len(y_true)}) and predicted labels ({len(y_pred)}).")
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    print("\n--- Classification Report ---")
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
        print(report)
    except ValueError as e:
        print(f"Error generating classification report: {e}")

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    plot_confusion_matrix(cm, class_names=class_names, save_path=CONFUSION_MATRIX_PLOT_PATH)

    print(f"\nBest trained model saved to {os.path.abspath(MODEL_SAVE_PATH)}")
    print("Evaluation complete.")

# --- Main execution block ---
if __name__ == "__main__":
    train_and_evaluate_emotion_model()