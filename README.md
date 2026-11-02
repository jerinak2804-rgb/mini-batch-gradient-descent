# Mini-Batch Gradient Descent for Linear Regression

A beginner-friendly Python demonstration of **Mini-Batch Gradient Descent (MBGD)**.
The program creates a small synthetic dataset, trains a linear regression model,
prints the learned parameters and loss, and displays two graphs.

## 1. Requirements

- Python 3.9 or newer
- NumPy
- Matplotlib

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the program

```bash
python mini_batch_gd.py
```

## 4. What the program demonstrates

1. Creates sample data following an approximately linear relationship.
2. Shuffles the dataset at the start of each epoch.
3. Divides the data into mini-batches of 32 samples.
4. Calculates gradients using each mini-batch.
5. Updates the weight and bias.
6. Prints the final weight, bias, and mean squared error.
7. Displays the training-loss curve and fitted regression line.

## 5. Main idea

- **Batch Gradient Descent:** uses the complete dataset for each update.
- **Stochastic Gradient Descent:** uses one training example for each update.
- **Mini-Batch Gradient Descent:** uses a small group of examples for each update.

Mini-batch gradient descent aims to combine the stable updates of batch gradient
descent with the faster, more frequent updates of stochastic gradient descent.
