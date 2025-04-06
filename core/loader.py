import torch
import idx2numpy
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from config import gconfig


def load_MNIST():
    """
    Load MNIST dataset and split into training, validation, and test sets.
    Note that the data folder is assumed to one level up from the current directory.
    :return: train_loader, val_loader, test_loader
    :return:
    """
    # Load raw MNIST data
    train_images = idx2numpy.convert_from_file("../" + gconfig.mnist_train_images_path)
    train_labels = idx2numpy.convert_from_file("../" + gconfig.mnist_train_labels_path)
    test_images = idx2numpy.convert_from_file("../" + gconfig.mnist_test_images_path)
    test_labels = idx2numpy.convert_from_file("../" + gconfig.mnist_test_labels_path)

    train_shape = train_images.shape  # (60000, 28, 28)
    test_shape = test_images.shape  # (10000, 28, 28)

    # Sanity checks
    assert train_shape == (60000, 28, 28)
    assert test_shape == (10000, 28, 28)

    # Split into training (50,000) and validation (10,000)
    train_images, val_images = train_images[:50000], train_images[50000:]
    train_labels, val_labels = train_labels[:50000], train_labels[50000:]

    # Convert to PyTorch tensors
    train_images = torch.tensor(train_images, dtype=torch.float32)
    val_images = torch.tensor(val_images, dtype=torch.float32)
    test_images = torch.tensor(test_images, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Normalize pixel values from [0, 255] to [0, 1]
    transform = transforms.Normalize((0,), (255,))  # Mean=0, Std=255 scales to [0, 1]
    train_images = train_images.unsqueeze(1)  # Add channel dim: (N, 1, 28, 28)
    val_images = val_images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)
    train_images = transform(train_images)
    val_images = transform(val_images)
    test_images = transform(test_images)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    # Create DataLoaders with shuffling for training
    train_loader = DataLoader(train_dataset, batch_size=gconfig.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=gconfig.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=gconfig.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_MNIST_Fashion():
    """
    Load Fashion MNIST dataset and split into training, validation, and test sets.
    Note that the data folder is assumed to one level up from the current directory.
    :return: train_loader, val_loader, test_loader
    :return:
    """
    # Load raw Fashion MNIST data
    train_images = idx2numpy.convert_from_file("../" + gconfig.fashion_train_images_path)
    train_labels = idx2numpy.convert_from_file("../" + gconfig.fashion_train_labels_path)
    test_images = idx2numpy.convert_from_file("../" + gconfig.fashion_test_images_path)
    test_labels = idx2numpy.convert_from_file("../" + gconfig.fashion_test_labels_path)

    train_shape = train_images.shape  # (60000, 28, 28)
    test_shape = test_images.shape  # (10000, 28, 28)

    # Sanity checks
    assert train_shape == (60000, 28, 28)
    assert test_shape == (10000, 28, 28)

    # Split into training (50,000) and validation (10,000)
    train_images, val_images = train_images[:50000], train_images[50000:]
    train_labels, val_labels = train_labels[:50000], train_labels[50000:]

    # Convert to PyTorch tensors
    train_images = torch.tensor(train_images, dtype=torch.float32)
    val_images = torch.tensor(val_images, dtype=torch.float32)
    test_images = torch.tensor(test_images, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # Normalize pixel values from [0, 255] to [0, 1]
    transform = transforms.Normalize((0,), (255,))  # Mean=0, Std=255 scales to [0, 1]
    train_images = train_images.unsqueeze(1)  # Add channel dim: (N, 1, 28, 28)
    val_images = val_images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)

    train_images = transform(train_images)
    val_images = transform(val_images)
    test_images = transform(test_images)
    # Create TensorDatasets
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    # Create DataLoaders with shuffling for training
    train_loader = DataLoader(train_dataset, batch_size=gconfig.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=gconfig.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=gconfig.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

