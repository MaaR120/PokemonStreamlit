import os
import random
from io import BytesIO

import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

TRADUCCION_TIPOS = {
    "Bug": "Bicho",
    "Dragon": "Dragon",
    "Electric": "Electrico",
    "Fairy": "Hada",
    "Fighting": "Lucha",
    "Fire": "Fuego",
    "Ghost": "Fantasma",
    "Grass": "Planta",
    "Ground": "Tierra",
    "Ice": "Hielo",
    "Normal": "Normal",
    "Poison": "Veneno",
    "Psychic": "Psiquico",
    "Rock": "Roca",
    "Water": "Agua",
}

CLASSES = list(TRADUCCION_TIPOS.values())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def obtener_transformacion():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def cargar_modelo_ganador(ruta_pesos):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(CLASSES))

    if os.path.exists(ruta_pesos):
        checkpoint = torch.load(ruta_pesos, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            epoch = checkpoint.get("epoch", "N/A")
        else:
            model.load_state_dict(checkpoint)
            epoch = "N/A"
        model.to(DEVICE)
        model.eval()
        return model, f"OK: pesos cargados (epoca {epoch})", True

    return None, "No se encontraron los pesos", False


def predecir_imagen(modelo, imagen_pil):
    """Procesa la imagen PIL y retorna la prediccion de la IA."""
    transform = obtener_transformacion()
    imagen_tensor = transform(imagen_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = modelo(imagen_tensor)
        probabilidades = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_idx = outputs.argmax(dim=1).item()

    return pred_idx, probabilidades, CLASSES


def obtener_pokemon_por_id(id_pokemon):
    """Trae un Pokemon de la Gen 1, su imagen y su tipo principal."""
    url = f"https://pokeapi.co/api/v2/pokemon/{id_pokemon}"

    try:
        respuesta = requests.get(url, timeout=20).json()
        nombre = respuesta["name"].capitalize()
        tipo_ingles = respuesta["types"][0]["type"]["name"].capitalize()
        tipo_real = TRADUCCION_TIPOS.get(tipo_ingles, "Normal")
        url_imagen = respuesta["sprites"]["other"]["official-artwork"]["front_default"]

        res_img = requests.get(url_imagen, timeout=20)
        imagen_pil = Image.open(BytesIO(res_img.content)).convert("RGB")

        return {
            "id": id_pokemon,
            "nombre": nombre,
            "tipo_real": tipo_real,
            "imagen": imagen_pil,
        }
    except Exception:
        return {
            "id": id_pokemon,
            "nombre": "Pikachu",
            "tipo_real": "Electrico",
            "imagen": Image.new("RGB", (224, 224), color="yellow"),
        }


def obtener_pokemon_aleatorio():
    """Trae un Pokemon aleatorio de la Gen 1."""
    return obtener_pokemon_por_id(random.randint(1, 151))


def obtener_lote_pokemon(cantidad=20):
    """Genera un lote de Pokemon unicos para jugar en modo masivo."""
    cantidad = max(1, min(cantidad, 20))
    ids = random.sample(range(1, 152), cantidad)
    return [obtener_pokemon_por_id(id_pokemon) for id_pokemon in ids]
