import numpy as np
import matplotlib.pyplot as plt

def graph(X, y, title=""):
    plt.scatter(X.flatten(), y.flatten(), marker=".")
    plt.title(title)
    plt.show()

def graph_line(X, y, w, b):
    plt.plot(X.flatten(), X @ w + b, color="red")
    plt.scatter(X.flatten(), y.flatten(), marker=".")
    plt.title(f"Linear Regression Results: y = {np.round(w.flatten()[0], 2)}x + {np.round(b.flatten()[0], 2)}")
    plt.show()

def graph_losses(losses):
    epochs = [i for i in range(len(losses))]
    plt.plot(epochs, losses)
    plt.title("Losses vs. Epochs")
    plt.show()