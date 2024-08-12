import torch
from timm import create_model
from torchvision import transforms
from PIL import Image

def load_model(checkpoint_path, model_name='resnet50'):
    try:
        model = create_model(model_name, pretrained=False, num_classes=2)  # Adjust num_classes based on your model
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

def preprocess_data(data_path):
    # Assuming the input data is in image form. If it's tabular or another format, adjust preprocessing accordingly.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data = Image.open(data_path).convert("RGB")
    data = transform(data)
    data = data.unsqueeze(0)  # Add batch dimension
    return data

def predict_outcome(model, data_tensor):
    if model is None:
        print("Model loading failed, cannot make prediction.")
        return None
    
    with torch.no_grad():
        outputs = model(data_tensor)
        _, predicted = outputs.max(1)
        return predicted.item()

if __name__ == "__main__":
    checkpoint_path = 'C:/TEJA/Results/checkpoint-best.pth'
    data_path = 'C:/TEJA/DataSet/2663.jpg'  # Replace with the correct path to your input data

    # Adjust 'model_name' to match the model used in the original training if known
    model = load_model(checkpoint_path, model_name='resnet50')
    
    if model:
        data_tensor = preprocess_data(data_path)
        prediction = predict_outcome(model, data_tensor)

        if prediction == 1:
            print("Prediction: The person is likely to die within the next 6 months.")
        else:
            print("Prediction: The person is unlikely to die within the next 6 months.")
