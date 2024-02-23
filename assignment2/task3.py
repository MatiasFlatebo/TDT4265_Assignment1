import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer
import numpy as np

def plot_accuracies(histories, labels, train_histories):
    plt.figure(figsize=(14, 7))

    for i in range(len(histories)):
        # Ensure history is a list of accuracies
        sorted_keys = sorted(histories[i].keys())
        sorted_values = [histories[i][key] for key in sorted_keys]
        sorted_keys_train = sorted(train_histories[i].keys())
        sorted_values_train = [train_histories[i][key] for key in sorted_keys_train]
        plt.plot(sorted_keys, sorted_values, label=labels[i])
        plt.plot(sorted_keys_train, sorted_values_train, label=labels[i] + " train")
    plt.title("Validation Accuracy Across Different Scenarios")
    plt.xlabel("Epoch")
    plt.ylim([0.6, 1.1])
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("task4d_accuracies_with_diff_tricks.png")
    plt.show()

def plot_loss(histories, labels, train_histories):
    plt.figure(figsize=(14, 7))

    for i in range(len(histories)):
        # Ensure history is a list of accuracies
        sorted_keys = sorted(histories[i].keys())
        sorted_values = [histories[i][key] for key in sorted_keys]
        sorted_keys_train = sorted(train_histories[i].keys())
        sorted_values_train = [train_histories[i][key] for key in sorted_keys_train]
        plt.plot(sorted_keys_train, sorted_values_train, label=labels[i] + " train")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylim([0, 3])
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("task4e_loss_with_diff_tricks.png")
    plt.show()

def main():
    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    x_mean, x_std = np.mean(X_train), np.std(X_train)
    X_train = pre_process_images(X_train, x_mean, x_std)
    X_val = pre_process_images(X_val, x_mean, x_std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    scenarios = [
        (False, False, False, "Original"),
        (True, False, False, "Improved Sigmoid"),
        (False, True, False, "Improved Weight Init"),
        (False, False, True, "With Momentum"),
        (True, True, True, "All Improvements"),
    ]

    num_epochs = 50
    learning_rate = 0.02
    batch_size = 32
    momentum_gamma = 0.9
    shuffle_data = True

    architecture = {
        "Task 2and3" : [64, 10],
        "Task 4a" : [32, 10],
        "Task 4b" : [128, 10],
        "Task 4d" : [59, 59, 10], #Må være like mange params
        "Task 4e" : [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10],
    }

    histories = []
    train_histories = []
    labels = []
    for use_improved_sigmoid, use_improved_weight_init, use_momentum, label in [scenarios[-1]]:
        for i in range(2):
            if i == 0:
                model = SoftmaxModel(architecture["Task 2and3"], use_improved_sigmoid, use_improved_weight_init, False)
            else:
                model = SoftmaxModel(architecture["Task 4e"], use_improved_sigmoid, use_improved_weight_init, False)
            trainer = SoftmaxTrainer(
                momentum_gamma, use_momentum, model, learning_rate, batch_size,
                shuffle_data, X_train, Y_train, X_val, Y_val
            )
            train_history, val_history = trainer.train(num_epochs)
            # Append specific accuracy history for plotting
            train_histories.append(train_history["loss"])
            histories.append(val_history["loss"])
            labels.append(label)

    for i in range(len(histories)):
        max_key = min(key for key in histories[i].keys())
        print(f"Min Loss for scenario {labels[i]}", histories[i][max_key])
    plot_loss(histories, ["Baseline model", "Deep model"], train_histories)

if __name__ == "__main__":
    main()
