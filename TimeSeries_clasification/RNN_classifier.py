import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import random
import tempfile
import os
import matplotlib.pyplot as plt
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import LSTM

# -----------------------------
# Step 1: Fix seeds for reproducibility
# -----------------------------
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# -----------------------------
# Step 2: Simulate example data with n conditions
# -----------------------------
def simulate_data(n_conditions=3, n_trials=100, n_timesteps=200):
    conditions = []
    for i in range(n_conditions):
        base = np.sin(np.linspace(0, 2*np.pi*(i+1), n_timesteps))
        cond = np.array([
            base + 0.2*np.random.randn(n_timesteps)
            for _ in range(n_trials)
        ])
        conditions.append(cond)
    return np.array(conditions)  # shape (n_conditions, n_trials, n_timesteps)

# -----------------------------
# Step 3: Prepare data
# -----------------------------
def prepare_data(data):
    n_conditions, n_trials, n_timesteps = data.shape
    X = data.reshape(-1, n_timesteps)
    y = np.repeat(np.arange(n_conditions), n_trials)
    # Add channel dimension for Conv1D
    X = X[..., np.newaxis]
    return X, y


def preprocess_data(X):
    # Normalize per trial
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    X = X[..., np.newaxis] # add channel dimension for Conv1D
    return X

# -----------------------------
# Step 4: Build classifier (Conv1D + BatchNorm)
# -----------------------------
def build_model(n_timesteps, n_classes,learning_rate=1e-3,dropout_rate=0.3, l2_reg=1e-2):
    model = Sequential([
        Conv1D(4, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(l2_reg),
               input_shape=(n_timesteps, 1)),
        BatchNormalization(),
        Conv1D(8, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        LSTM(8, return_sequences=False, kernel_regularizer=regularizers.l2(l2_reg)),
        Dropout(dropout_rate),
        Dense(8, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        Dense(n_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------
# Step 5: Cross-validation & significance test
# -----------------------------
def train_model(X, y, epochs=50):
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)


    model = build_model(X.shape[1], len(np.unique(y)))
    history = model.fit(X, y, validation_split=0.2, epochs=epochs, batch_size=16,
    shuffle=True, callbacks=[rlrop, early_stop], verbose=1)


    # Plot accuracy and loss
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(history.history['accuracy'], label='Train Acc')
    ax[0].plot(history.history['val_accuracy'], label='Val Acc')
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()


    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()


    fig.tight_layout()
    fig.show()


    return model


# -----------------------------
# Step 6: Permutation test
# -----------------------------
def permutation_test(X, y, model_builder, n_permutations=100, epochs=50):
    observed_model = model_builder(X.shape[1], len(np.unique(y)))
    observed_model.fit(X, y, epochs=epochs, batch_size=16, shuffle=True, verbose=0)
    y_pred = np.argmax(observed_model.predict(X, verbose=0), axis=1)
    observed_acc = accuracy_score(y, y_pred)


    null_accs = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_model = model_builder(X.shape[1], len(np.unique(y)))
        perm_model.fit(X, y_perm, epochs=epochs, batch_size=16, shuffle=True, verbose=0)
        y_pred_perm = np.argmax(perm_model.predict(X, verbose=0), axis=1)
        null_accs.append(accuracy_score(y_perm, y_pred_perm))


    p_value = (np.sum(np.array(null_accs) >= observed_acc) + 1) / (n_permutations + 1)
    return observed_acc, null_accs, p_value

# -----------------------------
# Step 6: Run
# -----------------------------
if __name__ == "__main__":
    pass