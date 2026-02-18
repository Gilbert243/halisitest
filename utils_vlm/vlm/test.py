"""
Exemple d'utilisation du module utils_vlm

IMPORTANT: Remplacez les chemins d'images par vos propres fichiers avant d'exécuter.
Les credentials Azure OpenAI sont chargés depuis les variables d'environnement:
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
"""

import sys
import os

# Ajouter le dossier parent au path pour importer utils_vlm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils_vlm import generate_mask, predict_hair_type_from_mask, process_image_full_pipeline
import cv2

# Exemple avec une image (à remplacer par votre propre image)
IMAGE_PATH = "../vlm/data/new_3C_13.jpeg"  # Modifiez ce chemin

print("\n" + "=" * 60)
print("TEST DE l'ENVIRONNEMENT ET DES FONCTIONS")
print("=" * 60)
print("\nVérification des variables d'environnement pour Azure OpenAI...")
if not os.getenv("AZUREOPENAI_API_KEY") or not os.getenv("AZUREOPENAI_API_ENDPOINT"):
    print("ERREUR: Les variables d'environnement AZUREOPENAI_API_KEY et AZUREOPENAI_ENDPOINT ne sont pas définies.")
    print("Veuillez les définir avant d'exécuter ce test.")
    exit(1)
print("✓ Variables d'environnement trouvées.")

# Vérifier si l'image existe
if not os.path.exists(IMAGE_PATH):
    print(f"ERREUR: L'image '{IMAGE_PATH}' n'existe pas.")
    print("Veuillez modifier IMAGE_PATH dans test.py avec le chemin d'une vraie image.")
    print("\nExemple d'utilisation:")
    print("-" * 50)
    print("# Option 1: Utiliser les fonctions séparément")
    print("masked_image = generate_mask('votre_image.jpg')")
    print("cv2.imwrite('mask_output.jpg', masked_image)")
    print()
    print("result = predict_hair_type_from_mask('mask_output.jpg')")
    print("print(f'Type de cheveux: {result[\"hair_type\"]}')")
    print()
    print("# Option 2: Pipeline complet")
    print("result = process_image_full_pipeline(")
    print("    'votre_image.jpg',")
    print("    save_mask=True,")
    print("    mask_output_path='mask_saved.jpg'")
    print(")")
    print("print(f'Type détecté: {result[\"hair_type\"][\"hair_type\"]}')")
    exit(1)

# Option 1: Utiliser les fonctions séparément
print("=" * 60)
print("Option 1: Génération du masque et prédiction séparées")
print("=" * 60)

# Générer le masque
print("\n1. Génération du masque...")
masked_image = generate_mask(IMAGE_PATH)
cv2.imwrite("mask_output.jpg", masked_image)
print("   ✓ Masque sauvegardé: mask_output.jpg")

# Prédire le type de cheveux
print("\n2. Prédiction du type de cheveux...")
result = predict_hair_type_from_mask("mask_output.jpg")
print(f"   ✓ Type de cheveux détecté: {result['hair_type']}")

# Option 2: Pipeline complet
print("\n" + "=" * 60)
print("Option 2: Pipeline complet")
print("=" * 60)

result = process_image_full_pipeline(
    IMAGE_PATH,
    save_mask=True,
    mask_output_path="mask_saved.jpg",

)
print(f"\n✓ Type détecté: {result['hair_type']['hair_type']}")
print(f"✓ Masque sauvegardé: mask_saved.jpg")
