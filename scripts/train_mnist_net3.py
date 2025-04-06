import torch
import config.gconfig
from core.loader import load_MNIST
from core.train import train_model, evaluate_model
from core.models import Net3


def main():
    """
    Run train_mnist_net3.py to train the model.
    :return:
    """

    # Load MNIST data
    train_loader, val_loader, test_loader = load_MNIST()

    # Initialize the model
    net = Net3().to(config.gconfig.device)

    print("Model on device:", config.gconfig.device)

    # Initialize model
    net = train_model(net, train_loader, val_loader)

    # Evaluate the model
    evaluate_model(net, test_loader)

    # Save the model
    save_model_path = config.gconfig.build_dir + "mnist_net3_model.pth"
    torch.save(net.state_dict(), "../" + save_model_path)

    print(f"Training complete. {save_model_path}'.")


if __name__ == '__main__':
    main()