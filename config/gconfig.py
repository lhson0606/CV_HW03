import torch

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "data/"
mnist_train_images_path = f"{data_dir}MNIST/train-images.idx3-ubyte"
mnist_train_labels_path = f"{data_dir}MNIST/train-labels.idx1-ubyte"
mnist_test_images_path = f"{data_dir}MNIST/t10k-images.idx3-ubyte"
mnist_test_labels_path = f"{data_dir}MNIST/t10k-labels.idx1-ubyte"

fashion_train_images_path = f"{data_dir}FashionMNIST/train-images.idx3-ubyte"
fashion_train_labels_path = f"{data_dir}FashionMNIST/train-labels.idx1-ubyte"
fashion_test_images_path = f"{data_dir}FashionMNIST/t10k-images.idx3-ubyte"
fashion_test_labels_path = f"{data_dir}FashionMNIST/t10k-labels.idx1-ubyte"

build_dir = "build/"

learning_rate = 1e-2
num_epochs = 16
