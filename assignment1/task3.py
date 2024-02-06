import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    outputs = model.forward(X)
    
    predicted_labels = np.argmax(outputs, axis=1)
    actual_labels = np.argmax(targets, axis=1)

    correct_predictions = np.sum(predicted_labels == actual_labels)
    accuracy = correct_predictions / len(X)

    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        outputs = self.model.forward(X_batch)
        self.model.backward(X_batch, outputs, Y_batch)
        self.model.w = self.model.w - self.learning_rate * self.model.grad
        loss = cross_entropy_loss(Y_batch, outputs)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.grid()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    # Plot the weights with L2 reg using lamdba 0.0 and 1.0

    def plot_weights(model, title):
        weight_matrix = model.w[:-1].reshape(28, 28, 10) 
        figure, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i, ax in enumerate(axes):
            ax.imshow(weight_matrix[:, :, i], cmap='gray')
            ax.axis('off')
            ax.set_title(f"Digit {i}")
        plt.suptitle(title)
        plt.savefig(f"task4b_{title}.png")
        plt.show()

    train_histories = {}
    val_histories = {}


    lambdas = [1, 0.1, 0.01, 0.001, 0.0]
    models = [SoftmaxModel(l2_reg_lambda=i) for i in lambdas]

    for i, model in enumerate(models):
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        train_histories[lambdas[i]] = train_history
        val_histories[lambdas[i]] = val_history

    # Plot weights "density" for each digit
    plot_weights(models[-1], "Weights without regularization")
    plot_weights(models[0], "Weights with full regularization")

    for lambda_value in lambdas:
        utils.plot_loss(val_histories[lambda_value]["accuracy"], f"Validation Accuracy (lambda={lambda_value})")
    plt.ylim([0.75, 0.95])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Plotting of the l2 norm for each weight (Task 4e)
    l2_norms = [np.linalg.norm(model.w) for model in models]
    plt.plot(lambdas, l2_norms, '-o')
    plt.xlabel("Lambda")
    plt.ylabel("L2 Norm of Weights")
    plt.title("L2 Norm of Weights vs. Lambda")
    plt.grid()
    plt.savefig("task4d_l2_norms.png")
    plt.show()


if __name__ == "__main__":
    main()
