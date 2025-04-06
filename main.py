import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import idx2numpy
import torchvision.transforms as transforms
from core.models import Net4
import matplotlib.pyplot as plt


# switch matplotlib backend to TkAgg
plt.switch_backend('TkAgg')


model = Net4()
model.load_state_dict(torch.load("build/mnist_net4_model.pth"))
model.eval()


# GUI
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess image (assuming 28x28 grayscale)
        img = Image.open(file_path).convert("L").resize((28, 28))
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # (1, 1, 28, 28)
        # img_tensor = transforms.Normalize((0,), (255,))(img_tensor)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()

        # show img_tensor using matplotlib
        plt.imshow(img_tensor.squeeze(0).squeeze(0), cmap='gray')
        plt.title(f"Predicted: {pred}")
        plt.axis('off')
        plt.show()


# Tkinter window
root = tk.Tk()
root.title("MNIST Classifier")

btn = tk.Button(root, text="Load Image", command=load_image)
btn.pack()

root.mainloop()