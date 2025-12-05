# Exposes essential components of the library so they can be imported directly

from .layers import Dense
from .activations import ReLU, Sigmoid, Tanh, Softmax
from .losses import MSE
from .optimizers import SGD
from .network import Network
from .visualize import plot_losses, plot_decision_boundary
from .preprocessing import StandardScaler, shuffle_data, train_test_split