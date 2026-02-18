import os
import cv2
import json
import torch
import numpy as np
import yaml
from yaml.loader import SafeLoader
from PIL import Image
import sys
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import PetModel
from client import AzureLLMClient
from utils import encode_image

load_dotenv()

def generate_mask(image_path: str, model_path: str = None, config_path: str = None):
    """Génère un masque de cheveux à partir d'une image avec fond noir.
    
    Prend une image en entrée, applique un modèle de segmentation pour détecter
    les cheveux et retourne une image masquée où seuls les cheveux sont visibles
    sur un fond noir.
    
    Args:
        image_path: Chemin vers l'image d'entrée
        model_path: Chemin vers le modèle de segmentation (optionnel)
                   Par défaut: '../models/model_mobilenetv2_150_320x320.pth'
        config_path: Chemin vers le fichier de configuration (optionnel)
                    Par défaut: '../config/config.yaml'
    
    Returns:
        numpy.ndarray: Image masquée (BGR) avec fond noir et cheveux visibles
        
    Raises:
        FileNotFoundError: Si l'image, le modèle ou la config n'existent pas
        ValueError: Si l'image ne peut pas être chargée
        
    Example:
        >>> masked_img = generate_mask("photo.jpg")
        >>> cv2.imwrite("masked_output.jpg", masked_img)
    """
    # Chemins par défaut
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 
                                  'model_mobilenetv2_150_320x320.pth')
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 
                                   'config.yaml')
    
    # Vérifier que les fichiers existent
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    # Charger la configuration
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    # Charger le modèle
    model = PetModel(
        arch=config['HParams']['architecture'],
        encoder_name=config['HParams']['backbone'],
        in_channels=config['HParams']['nr_channels'],
        out_classes=config['HParams']['nr_identities']
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Préparer l'image pour l'inférence
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (320, 320), interpolation=cv2.INTER_AREA)
    
    # Convertir en tensor
    tensor = np.moveaxis(image_resized, -1, 0)
    tensor = torch.unsqueeze(torch.tensor(tensor, dtype=torch.float32), 0)
    
    # Prédire le masque
    with torch.no_grad():
        logits = model(tensor)
        mask = logits.sigmoid()
    
    mask_np = mask.squeeze().cpu().numpy()
    
    # Redimensionner le masque à la taille originale
    mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))
    
    # Créer un masque binaire (seuil à 0.5)
    binary_mask = (mask_resized > 0.5).astype(np.uint8)
    
    # Appliquer le masque - fond noir, cheveux visibles
    masked_image = image.copy()
    masked_image[binary_mask == 0] = 0
    
    return masked_image


def predict_hair_type_from_mask(mask_image_path: str, 
                                azure_api_key: str = os.getenv("AZUREOPENAI_API_KEY"),
                                azure_endpoint: str = os.getenv("AZUREOPENAI_API_ENDPOINT"),
                                azure_api_version: str = "2024-02-15-preview",
                                model_deployment: str = "gpt-4o-mini",
                                prompt_path: str = None,
                                temperature: float = 0,
                                timeout: int = 30):
    """Prédit le type de cheveux à partir d'une image de masque.
    
    Utilise Azure OpenAI Vision Model avec le prompt vision_only_6 pour
    classifier le type de cheveux visible dans le masque (1A-4C).
    
    Args:
        mask_image_path: Chemin vers l'image du masque (fond noir, cheveux visibles)
        azure_api_key: Clé API Azure OpenAI (ou variable d'env AZURE_OPENAI_API_KEY)
        azure_endpoint: Endpoint Azure OpenAI (ou variable d'env AZURE_OPENAI_ENDPOINT)
        azure_api_version: Version de l'API Azure (défaut: "2024-02-15-preview")
        model_deployment: Nom du déploiement du modèle (défaut: "gpt-4o-mini")
        prompt_path: Chemin vers le fichier prompt (optionnel)
                    Par défaut: '../prompts/vision_only_6.txt'
        temperature: Température de sampling (défaut: 0 pour déterministe)
        timeout: Timeout de la requête en secondes (défaut: 30)
    
    Returns:
        dict: Résultat de classification avec la clé 'hair_type'
              Exemple: {"hair_type": "3B"}
        
    Raises:
        FileNotFoundError: Si l'image ou le prompt n'existent pas
        ValueError: Si les credentials Azure ne sont pas fournis
        Exception: En cas d'erreur API (réseau, timeout, rate limit, etc.)
        
    Example:
        >>> result = predict_hair_type_from_mask(
        ...     "masked_hair.jpg",
        ...     azure_api_key="your-key",
        ...     azure_endpoint="https://your-resource.openai.azure.com/"
        ... )
        >>> print(result["hair_type"])  # "3B"
    """
    # Chemin par défaut pour le prompt
    if prompt_path is None:
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', 
                                   'vision_only_6.txt')
    
    # Vérifier que les fichiers existent
    if not os.path.exists(mask_image_path):
        raise FileNotFoundError(f"Mask image not found: {mask_image_path}")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    # Gérer les credentials Azure
    if azure_api_key is None:
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if azure_endpoint is None:
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    
    if not azure_api_key or not azure_endpoint:
        raise ValueError(
            "Azure OpenAI credentials required. Provide 'azure_api_key' and "
            "'azure_endpoint' arguments or set AZURE_OPENAI_API_KEY and "
            "AZURE_OPENAI_ENDPOINT environment variables."
        )
    
    # Charger le prompt système
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    
    # Encoder l'image en base64
    image_b64 = encode_image(mask_image_path)
    
    # Initialiser le client Azure
    client = AzureLLMClient(
        api_key=azure_api_key,
        endpoint=azure_endpoint,
        api_version=azure_api_version
    )
    
    # Effectuer la classification
    response_json = client.classify(
        model=model_deployment,
        system_prompt=system_prompt,
        image_b64=image_b64,
        temperature=temperature,
        timeout=timeout
    )
    
    # Parser la réponse JSON
    result = json.loads(response_json)
    
    return result


def process_image_full_pipeline(image_path: str,
                                azure_api_key: str = os.getenv("AZUREOPENAI_API_KEY"),
                                azure_endpoint: str = os.getenv("AZUREOPENAI_API_ENDPOINT"),
                                model_path: str = None,
                                config_path: str = None,
                                prompt_path: str = None,
                                save_mask: bool = False,
                                mask_output_path: str = None):
    """Pipeline complet: génère le masque et prédit le type de cheveux.
    
    Fonction de commodité qui enchaîne les deux opérations:
    1. Génération du masque à partir de l'image
    2. Prédiction du type de cheveux à partir du masque
    
    Args:
        image_path: Chemin vers l'image d'entrée
        azure_api_key: Clé API Azure OpenAI
        azure_endpoint: Endpoint Azure OpenAI
        model_path: Chemin vers le modèle de segmentation (optionnel)
        config_path: Chemin vers la configuration (optionnel)
        prompt_path: Chemin vers le prompt (optionnel)
        save_mask: Si True, sauvegarde le masque généré (défaut: False)
        mask_output_path: Chemin de sortie du masque (requis si save_mask=True)
    
    Returns:
        dict: Dictionnaire avec deux clés:
              - 'mask': numpy.ndarray de l'image masquée
              - 'hair_type': dict avec la prédiction
              
    Example:
        >>> result = process_image_full_pipeline(
        ...     "photo.jpg",
        ...     azure_api_key="key",
        ...     azure_endpoint="https://...",
        ...     save_mask=True,
        ...     mask_output_path="output_mask.jpg"
        ... )
        >>> print(f"Type détecté: {result['hair_type']['hair_type']}")
    """
    import tempfile
    
    # Étape 1: Générer le masque
    print("Génération du masque...")
    masked_image = generate_mask(image_path, model_path, config_path)
    
    # Sauvegarder temporairement ou définitivement le masque
    if save_mask and mask_output_path:
        cv2.imwrite(mask_output_path, masked_image)
        temp_mask_path = mask_output_path
        print(f"Masque sauvegardé: {mask_output_path}")
    else:
        # Créer un fichier temporaire
        temp_fd, temp_mask_path = tempfile.mkstemp(suffix='.jpg')
        os.close(temp_fd)
        cv2.imwrite(temp_mask_path, masked_image)
    
    try:
        # Étape 2: Prédire le type de cheveux
        print("Prédiction du type de cheveux...")
        hair_type_result = predict_hair_type_from_mask(
            temp_mask_path,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            prompt_path=prompt_path
        )
        
        return {
            'mask': masked_image,
            'hair_type': hair_type_result
        }
    
    finally:
        # Nettoyer le fichier temporaire si nécessaire
        if not save_mask and os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)
