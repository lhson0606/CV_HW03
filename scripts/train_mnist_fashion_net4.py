import torch
import config.gconfig
from core.loader import load_MNIST_Fashion
from core.train import train_model, evaluate_model
from core.models import Net4


def main():
    """
    Run train_mnist_fashion_net4.py to train the model.
    :return:
    """

    # Load MNIST Fashion data
    train_loader, val_loader, test_loader = load_MNIST_Fashion()

    # Initialize the model
    net = Net4().to(config.gconfig.device)

    print("Model on device:", config.gconfig.device)

    # Initialize model
    net = train_model(net, train_loader, val_loader)

    # Evaluate the model
    evaluate_model(net, test_loader)

    # Save the model
    save_model_path = config.gconfig.build_dir + "mnist_fashion_net4_model.pth"
    torch.save(net.state_dict(), "../" + save_model_path)

    print(f"Training complete. {save_model_path}'.")


if __name__ == '__main__':
    main()