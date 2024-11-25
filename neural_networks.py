import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr 
        self.activation_fn = activation 
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))

        self.weights_output_hidden = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))

        self.activations = {'tanh': np.tanh, 'sigmoid': lambda x: 1 / (1 + np.exp(-x)), 'relu': lambda x: np.maximum(0, x)}
        self.activation_derivs = {
            'tanh': lambda x: 1 - np.tanh(x) ** 2,
            'sigmoid': lambda x: self.activations['sigmoid'](x) * (1 - self.activations['sigmoid'](x)),
            'relu': lambda x: (x > 0).astype(float)
        }

    def forward(self, X):
        self.b1 = X @ self.weights_input_hidden + self.bias_hidden
        self.a1 = self.activations[self.activation_fn](self.b1)
        self.b2 = self.a1 @ self.weights_output_hidden + self.bias_output
        out = self.b2
        return out

    def backward(self, X, y):
        output = self.forward(X)
        error = output - y
        grad_output = error
        grad_weights_output_hidden = self.a1.T @ grad_output
        grad_bias_output = np.sum(grad_output, axis=0, keepdims=True)

        grad_hidden = grad_output @ self.weights_output_hidden.T * self.activation_derivs[self.activation_fn](self.b1)
        grad_weights_input_hidden = X.T @ grad_hidden
        grad_bias_hidden = np.sum(grad_hidden, axis=0, keepdims=True)

        self.weights_input_hidden -= self.lr * grad_weights_input_hidden
        self.bias_hidden -= self.lr * grad_bias_hidden
        self.weights_output_hidden -= self.lr * grad_weights_output_hidden
        self.bias_output -= self.lr * grad_bias_output

        pass

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
        
    hidden_features = mlp.a1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)


    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid)
    preds = preds.reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], alpha=0.3, cmap='bwr')


    for i in range(mlp.weights_input_hidden.shape[0]):
        for j in range(mlp.weights_input_hidden.shape[1]):
            x_pos = [0, mlp.weights_input_hidden[i, j]]
            y_pos = [0, mlp.bias_hidden[0, j]]
            ax_gradient.plot(x_pos, y_pos, 'k-', alpha=0.5)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)