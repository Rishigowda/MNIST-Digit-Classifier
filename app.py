from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
from mnist_model import MNISTModel


# Initializing FastAPI app
app = FastAPI(
    title="MNIST Digit Classifier",
    description="This service classifies handwritten digits (0-9).\n Created by Hrishik B S for Applications of AI class.",
    version="1.0.0",
)

# Loading the trained model
model = MNISTModel()
checkpoint = torch.load("mnist_model.pth")  # Load the saved state dictionary
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Preprocessing pipeline same as training
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict the digit class for the uploaded image.
    
    Parameters:
    - file: Uploaded image file.
    
    Returns:
    - Predicted digit class.
    """
    try:
        # Loads the uploaded image
        image = Image.open(file.file).convert("L")  # Convert to grayscale
        tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Makes prediction
        with torch.no_grad():
            output = model(tensor.view(1, -1))  # Flatten input to match model
            print("Model raw output (logits):", output.numpy())  # Debugging logits
            prediction = torch.argmax(output, dim=1).item()

        return {"predicted_class": prediction}

    except Exception as e:
        return {"error": f"Error during prediction: {e}"}

@app.get("/")
async def home():
    """
    Home endpoint to test the service.
    """
    return {"message": "Welcome to MNIST Digit Classifier! Upload an image to predict and get started."}
