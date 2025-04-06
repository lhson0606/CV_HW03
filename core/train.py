import torch
from config import gconfig


def display_progress(epoch, num_epochs, running_loss, val_loss):
    """
    Display training progress.
    :param epoch: Current epoch
    :param num_epochs: Total number of epochs
    :param running_loss: Training loss
    :param val_loss: Validation loss
    :return:
    """
    print(f"Epoch [{epoch + 1}/{num_epochs}], "
          f"Training Loss: {running_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}")


def train_model(net, train_loader, val_loader):
    # We will use CrossEntropyLoss for multi-class classification
    criterion = torch.nn.CrossEntropyLoss()
    # Gradient Descent is too expensive for this task, so we will use Stochastic Gradient Descent instead
    optimizer = torch.optim.SGD(net.parameters(), lr=gconfig.learning_rate)

    # Training loop
    num_epochs = gconfig.num_epochs
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for images, labels in train_loader:
            # Move data to GPU if available
            images = images.to(gconfig.device)
            labels = labels.to(gconfig.device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation loss
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(gconfig.device)
                labels = labels.to(gconfig.device)

                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Display progress
        display_progress(epoch, num_epochs, running_loss / len(train_loader), val_loss / len(val_loader))

    return net


def evaluate_model(net, test_loader):
    """
    Evaluate the model on the test set.
    :param net: Trained model
    :param test_loader: DataLoader for the test set
    :return: Accuracy
    """

    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(gconfig.device)
            labels = labels.to(gconfig.device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
    return accuracy
