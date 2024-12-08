import torch
from torchvision import transforms
from PIL import Image
from mnist_model import MNISTModel
import matplotlib.pyplot as plt

# Loading the trained model
model = MNISTModel()
checkpoint = torch.load("mnist_model.pth")  # Load the saved state dictionary
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Preprocessing pipeline same as training and app.py
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28,28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,),(0.5,))  # Normalize to [-1, 1]
])

def test_model(image_path):
    """
    Test the model on a single image.

    Parameters:
    - image_path: Path to the image file.

    Returns:
    - Predicted digit class.
    """
    # Loading and preprocessing the image
    try:
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Debug: Visualizes the preprocessed image
        plt.imshow(tensor.squeeze(0).squeeze(0).numpy(), cmap="gray")
        plt.title("Preprocessed Image")
        plt.show()

        # Predicts using the model
        with torch.no_grad():
            output = model(tensor.view(1, -1))  # Flatten input to match model
            print("Model raw output (logits):", output.numpy())  # Debugging logits
            prediction = torch.argmax(output, dim=1).item()

        return prediction

    except Exception as e:
        print(f"Error during testing: {e}")
        return None

if __name__ == "__main__":
    # Tests with an example image
    image_path = "/Users/rishigowda/Downloads/download.png"  # Replace with your test image path
    predicted_class = test_model(image_path)
    print(f"Predicted class for the provided image: {predicted_class}")


