import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import requests
from PIL import Image
from io import BytesIO
import random

TRADUCCION_TIPOS = {
    'Bug': 'Bicho', 'Dragon': 'Dragón', 'Electric': 'Eléctrico', 'Fairy': 'Hada',
    'Fighting': 'Lucha', 'Fire': 'Fuego', 'Ghost': 'Fantasma', 'Grass': 'Planta',
    'Ground': 'Tierra', 'Ice': 'Hielo', 'Normal': 'Normal', 'Poison': 'Veneno',
    'Psychic': 'Psíquico', 'Rock': 'Roca', 'Water': 'Agua'
}

CLASSES = list(TRADUCCION_TIPOS.values())

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def obtener_transformacion():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def cargar_modelo_ganador(ruta_pesos):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASSES))

    if os.path.exists(ruta_pesos):
        checkpoint = torch.load(ruta_pesos, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'N/A')
        else:
            model.load_state_dict(checkpoint)
            epoch = "N/A"
        model.to(DEVICE)
        model.eval()
        return model, f"✅ Pesos cargados (Época {epoch})", True
    else:
        return None, f"❌ No se encontraron los pesos", False

def predecir_imagen(modelo, imagen_pil):
    """Procesa la imagen PIL y retorna la predicción de la IA."""
    transform = obtener_transformacion()
    imagen_tensor = transform(imagen_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = modelo(imagen_tensor)
        probabilidades = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = outputs.argmax(dim=1).item()

    return pred_idx, probabilidades, CLASSES

# --- NUEVAS FUNCIONES PARA EL JUEGO AUTOMÁTICO ---

def obtener_pokemon_aleatorio():
    """Trae un Pokémon aleatorio de la Gen 1, su imagen y su tipo principal formateado en español."""
    id_pokemon = random.randint(1, 151) # Generación 1
    url = f"https://pokeapi.co/api/v2/pokemon/{id_pokemon}"

    # Mapeo inverso para buscar por nombre en inglés y traducir
    tipo_to_es = TRADUCCION_TIPOS

    try:
        respuesta = requests.get(url).json()
        nombre = respuesta['name'].capitalize()

        # Conseguir el tipo principal en inglés
        tipo_ingles = respuesta['types'][0]['type']['name'].capitalize()

        # Traducir al español
        tipo_real = tipo_to_es.get(tipo_ingles, 'Normal')

        # URL de la imagen oficial (sprites de alta calidad)
        url_imagen = respuesta['sprites']['other']['official-artwork']['front_default']

        # Descargar la imagen y convertirla a PIL RGB
        res_img = requests.get(url_imagen)
        imagen_pil = Image.open(BytesIO(res_img.content)).convert("RGB")

        return nombre, tipo_real, imagen_pil
    except Exception as e:
        # Fallback seguro por si falla la conexión a internet
        return "Pikachu", "Eléctrico", Image.new('RGB', (224, 224), color='yellow')