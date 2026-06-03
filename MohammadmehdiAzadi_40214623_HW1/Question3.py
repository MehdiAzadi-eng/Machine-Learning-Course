import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from mpl_toolkits.mplot3d import Axes3D

def target_function(function_number, x, y):
    if function_number == 1:
        return 0.5 + 0.1 * (x**2) * np.cos(y + 3) + 0.4 * x * y * np.exp(1 - y**2)
    elif function_number == 2:
        term1 = np.exp(-(x**2))
        term2 = np.exp(-2*(y**2))
        term3 = 2 * np.exp(-0.5 * (x**2 + y**2))
        return np.maximum.reduce([term1, term2, term3])

range_val = 3.0
x = np.linspace(-range_val, range_val, 50)
y = np.linspace(-range_val, range_val, 50)
X, Y = np.meshgrid(x, y)

x_flat = X.flatten()
y_flat = Y.flatten()
inputs = np.column_stack((x_flat, y_flat))
targets = target_function(2, x_flat, y_flat)

model = Sequential([
    Dense(64, activation='relu', input_dim=2),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

print("learning progress...")
history = model.fit(inputs, targets, epochs=400, verbose=0, batch_size=32)
print("End of learning")

predictions = model.predict(inputs).flatten()
predictions_grid = predictions.reshape(X.shape)

error_grid = np.abs(targets.reshape(X.shape) - predictions_grid)

fig = plt.figure(figsize=(18, 5))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, targets.reshape(X.shape), cmap='viridis')
ax1.set_title('True Function')
ax1.set_xlabel('X') ; ax1.set_ylabel('Y')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, predictions_grid, cmap='plasma')
ax2.set_title('MLP Approximation')
ax2.set_xlabel('X') ; ax2.set_ylabel('Y')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, error_grid, cmap='Reds')
ax3.set_title('Absolute Error')
ax3.set_xlabel('X') ; ax3.set_ylabel('Y')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'])
plt.title('Model Loss (MSE) during Training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.show()
