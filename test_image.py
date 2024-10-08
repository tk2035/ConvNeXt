import torch
from timm import create_model
from torchvision import transforms
from PIL import Image

def load_model(checkpoint_path, model_name='resnet50'):
    try:
        model = create_model(model_name, pretrained=False, num_classes=2)  # num_classes adjusted based on your needs
    except RuntimeError as e:
        print(f"Error creating model: {e}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    try:
        # Loading the state_dict with strict=False to handle mismatches
        model.load_state_dict(checkpoint['model'], strict=False)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        return None
    
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_image(model, image_tensor):
    if model is None:
        print("Model loading failed, cannot make prediction.")
        return None
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        return predicted.item()

if __name__ == "__main__":
    checkpoint_path = 'C:/TEJA/Results/checkpoint-best.pth'
    image_path = 'C:/TEJA/DataSet/271.jpg'

    # Adjust 'model_name' to match the model used in the original training if known
    model = load_model(checkpoint_path, model_name='resnet50')
    
    if model:
        image_tensor = preprocess_image(image_path)
        prediction = predict_image(model, image_tensor)

        if prediction == 1:
            print("Prediction: COVID Positive")
        else:
            print("Prediction: COVID Negative")
