import os
import torch
import torch.nn as nn
from torchvision import models, transforms

# Lista de clases exacta en orden alfabético
CLASSES = ['Bug', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Ghost', 
           'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Water']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def obtener_transformacion():
    """Retorna el pipeline de preprocesamiento idéntico al de validación/test."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def cargar_modelo_ganador(ruta_pesos):
    """Crea la estructura de la ResNet18 y carga los pesos entrenados."""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASSES))
    
    if os.path.exists(ruta_pesos):
        # map_location asegura que cargue en CPU si Streamlit Cloud no tiene GPU
        checkpoint = torch.load(ruta_pesos, map_location=DEVICE)
        
        # Manejo por si guardaste el dict completo o solo el state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'N/A')
        else:
            model.load_state_dict(checkpoint)
            epoch = "N/A"
            
        model.to(DEVICE)
        model.eval()
        return model, f"✅ Pesos cargados con éxito (Época {epoch})", True
    else:
        return None, f"❌ No se encontraron los pesos en: {ruta_pesos}", False

def predecir_imagen(modelo, imagen):
    """Procesa la imagen y retorna las probabilidades de todas las clases y la predicción."""
    transform = obtener_transformacion()
    imagen_tensor = transform(imagen).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = modelo(imagen_tensor)
        probabilidades = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = outputs.argmax(dim=1).item()
        
    return pred_idx, probabilidades, CLASSES