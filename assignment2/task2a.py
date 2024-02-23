import numpy as np
import utils
import typing

np.random.seed(1)


def pre_process_images(X: np.ndarray, x_mean: float, x_std: float):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    X_norm = (X - x_mean) / x_std
    
    new_column = np.ones((X_norm.shape[0], 1))
    X_norm = np.hstack((X_norm, new_column))
    
    return X_norm


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    # TODO implement this function (Task 3a)
   
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
        
    return -np.sum(targets * np.log(outputs), axis=1).mean()


class SoftmaxModel:

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init
        self.hidden_layer_output = None

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

        # Set weights
        for i in range(len(self.ws)):
            if self.use_improved_weight_init:
                std = 1/np.sqrt(self.ws[i].shape[0])
                self.ws[i] = np.random.normal(0, std, size=self.ws[i].shape)
            else:
                self.ws[i] = np.random.uniform(-1, 1, size=self.ws[i].shape)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        self.hidden_layer_output = []
        prev_layer_vector = X

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def improved_sigmoid(x):
            return 1.7159 * np.tanh(2/3 * x)

        
        prev_layer_vector = X 
        for i in range(len(self.ws) -1):
            prev_layer_vector = prev_layer_vector @ self.ws[i]
            self.hidden_layer_output.append(prev_layer_vector)
            if self.use_improved_sigmoid:
                prev_layer_vector = improved_sigmoid(prev_layer_vector)
            else:
                prev_layer_vector = sigmoid(prev_layer_vector)

        z_k = np.dot(prev_layer_vector, self.ws[-1])
        output_layer_vector = np.exp(z_k) / np.sum(np.exp(z_k), axis=1, keepdims=True)
        return output_layer_vector
        
        

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
    
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_derivative(x):
            return sigmoid(x) * (1 - sigmoid(x))
        
        def improved_sigmoid_derivative(x):
            inserted = lambda x: 1 / np.cosh(x)
            return 1.7159 * 2/3 * (inserted((2/3) * x) ** 2)
        
        def improved_sigmoid(x):
            return 1.7159 * np.tanh(2/3 * x)
        
        # Needs to use self.hidden_layer_output to calculate the gradients
        self.grads = [None for _ in range(len(self.ws))]
        # Between last hidden layer and output layer
        self.grads[-1] = (sigmoid(self.hidden_layer_output[-1]).T @ (outputs-targets)) / X.shape[0]
        # For the hidden layers
        prev_grad = (outputs - targets)
        for i in range(len(self.ws)-2, -1, -1):
            updated_grad = prev_grad @ self.ws[i+1].T
            if self.use_improved_sigmoid:
                updated_grad = updated_grad * improved_sigmoid_derivative(self.hidden_layer_output[i])
            else:
                updated_grad = updated_grad * sigmoid_derivative(self.hidden_layer_output[i])

            layer_grad = None
            if i == 0:
                layer_grad = X.T @ updated_grad

            else:
                if self.use_improved_sigmoid:
                    layer_grad = improved_sigmoid(self.hidden_layer_output[i-1]).T @ updated_grad
                else:
                    layer_grad = sigmoid(self.hidden_layer_output[i-1]).T @ updated_grad
            self.grads[i] = layer_grad / X.shape[0]
            prev_grad = updated_grad
        

    #def zero_grad(self) -> None:
        #self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    one_hot_encoded = np.eye(num_classes)[np.squeeze(Y, axis=1)]
    return one_hot_encoded


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    x_mean, x_std  = np.mean(X_train), np.std(X_train)
    X_train = pre_process_images(X_train, x_mean, x_std)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_relu = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
