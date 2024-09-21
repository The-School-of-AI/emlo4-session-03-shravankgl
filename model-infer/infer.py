import json
import time
import random
import torch
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from model import Net  # Assuming the model is defined in model.py


def infer(model, dataset, save_dir, num_samples=5):
    model.eval()
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Randomly select `num_samples` from the dataset
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        image, _ = dataset[idx]  # Ignore the actual label

        # Perform inference
        with torch.no_grad():
            output = model(image.unsqueeze(0))  # Add batch dimension
        pred = output.argmax(dim=1, keepdim=True).item()  # Get predicted label

        # Convert the image to a format that can be saved (from tensor to PIL image)
        img = Image.fromarray(image.squeeze().numpy() * 255).convert("L")
        
        # Save the image with the predicted label as filename
        img.save(results_dir / f"{pred}.png")

        print(f"Saved image as {pred}.png in {results_dir}")


def main():
    # Directory where results will be saved (inside the mounted mnist volume)
    save_dir = "./model"  

    # Initialize the model and load checkpoint
    model = Net()  # Replace Net with your model's architecture
    model.load_state_dict(torch.load("./model/mnist_cnn.pt"))  # Load from the volume
    model.eval()  # Set model to evaluation mode

    # Create transformations for the MNIST dataset (normalize to [0,1] range)
    transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST test dataset (download=True ensures it is downloaded if not present)
    dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

    # Run inference on the dataset and save results
    infer(model, dataset, save_dir)
    print("Inference completed. Results saved in the 'results' folder inside the mnist volume.")


if __name__ == "__main__":
    main()
