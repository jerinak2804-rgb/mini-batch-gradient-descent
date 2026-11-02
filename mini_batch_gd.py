"""
Mini-Batch Gradient Descent for Linear Regression
Run: python mini_batch_gd.py
"""

import numpy as np
import matplotlib.pyplot as plt


def make_dataset(n_samples=1000, noise_std=15, seed=42):
    """Create a simple one-feature linear-regression dataset."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n_samples, 1))
    y = 3.0 * X[:, 0] + 2.0 + rng.normal(0, noise_std, size=n_samples)
    return X, y


def mini_batch_gradient_descent(X, y, learning_rate=0.01,
                                batch_size=32, epochs=100, seed=42):
    """Train y_hat = w*x + b using mini-batch gradient descent."""
    rng = np.random.default_rng(seed)
    n_samples = len(y)
    w = 0.0
    b = 0.0
    losses = []

    for epoch in range(epochs):
        # Shuffle the examples at the beginning of each epoch.
        indices = rng.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, n_samples, batch_size):
            X_batch = X_shuffled[start:start + batch_size, 0]
            y_batch = y_shuffled[start:start + batch_size]

            predictions = w * X_batch + b
            errors = predictions - y_batch

            # Gradients of mean squared error for this mini-batch.
            dw = (2 / len(X_batch)) * np.sum(errors * X_batch)
            db = (2 / len(X_batch)) * np.sum(errors)

            # Update model parameters.
            w -= learning_rate * dw
            b -= learning_rate * db

        # Record full-dataset loss after each epoch.
        full_predictions = w * X[:, 0] + b
        epoch_loss = np.mean((full_predictions - y) ** 2)
        losses.append(epoch_loss)

    return w, b, losses


def main():
    X, y = make_dataset()

    w, b, losses = mini_batch_gradient_descent(
        X, y,
        learning_rate=0.01,
        batch_size=32,
        epochs=100
    )

    final_predictions = w * X[:, 0] + b
    final_loss = np.mean((final_predictions - y) ** 2)

    print("Mini-Batch Gradient Descent: Linear Regression")
    print(f"Final weight (w): {w:.4f}")
    print(f"Final bias   (b): {b:.4f}")
    print(f"Final MSE loss  : {final_loss:.4f}")

    # Plot 1: training loss.
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss - Mini-Batch Gradient Descent")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 2: data and learned regression line.
    plt.figure(figsize=(7, 4))
    plt.scatter(X[:, 0], y, s=12, alpha=0.45, label="Training data")
    x_line = np.linspace(X.min(), X.max(), 200)
    plt.plot(x_line, w * x_line + b, linewidth=2, label="Predicted line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
