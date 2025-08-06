import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Generate dummy data for two yoga poses:
# Class 0 = Tree Pose, Class 1 = Warrior Pose
num_samples = 500

# Simulated landmark patterns
X_tree = np.random.rand(num_samples, 66) * 0.5 + 0.25     # Tree Pose around 0.5
X_warrior = np.random.rand(num_samples, 66) * 0.5 + 0.5   # Warrior Pose around 1.0

X = np.vstack((X_tree, X_warrior))
y = np.array([0]*num_samples + [1]*num_samples)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple feedforward neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(66,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 output classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("pose_model.h5")

print("✅ Model saved as pose_model.h5")
