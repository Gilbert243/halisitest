# Guide d'Utilisation - Notebook d'Analyse de Performance

## 📓 Nouveau Notebook `run.ipynb`

Le notebook a été complètement refait avec une structure claire et professionnelle.

## 🎯 Caractéristiques Principales

### 1. **Dataset Équilibré**
- Échantillonnage de **10 images par classe** pour éviter le biais dû au déséquilibre
- Comparaison entre dataset original et dataset équilibré
- Total: 100 images (10 classes × 10 images)

### 2. **Analyse Complète**
Le notebook contient 12 étapes:

1. **Import Libraries** - Configuration de l'environnement
2. **Load Configuration** - Chargement des paramètres API
3. **Original Dataset Analysis** - Analyse du dataset complet
4. **Balanced Dataset Creation** - Création du dataset équilibré (10/classe)
5. **System Initialization** - Configuration du classifier
6. **Run Evaluation** - Évaluation sur le dataset équilibré
7. **Overall Metrics** - Métriques globales (accuracy, F1, etc.)
8. **Confusion Matrix** - Matrices de confusion (absolue et normalisée)
9. **Per-Class Metrics** - Métriques détaillées par classe
10. **Latency Analysis** - Analyse de la latence et throughput
11. **Error Analysis** - Analyse des erreurs de classification
12. **Summary Report** - Rapport complet pour votre document

### 3. **Visualisations de Qualité**
- Graphiques en barres et camemberts pour la distribution
- Matrices de confusion colorées
- Graphiques de performance par classe
- Histogrammes et box plots de latence
- Heatmaps d'erreurs

## 🚀 Comment Utiliser

### Étape 1: Vérifier l'environnement

```bash
# Activer l'environnement virtuel
source env/bin/activate

# Vérifier que les variables d'environnement sont configurées
echo $AZUREOPENAI_API_KEY
echo $AZUREOPENAI_API_ENDPOINT
```

### Étape 2: Lancer le notebook

Ouvrez `run.ipynb` dans VS Code ou Jupyter et **exécutez les cellules séquentiellement**.

### Étape 3: Durée d'exécution

- **Cellules 1-5**: ~10 secondes (configuration)
- **Cellule 6**: ~3-5 minutes (évaluation de 100 images)
- **Cellules 7-12**: ~30 secondes (analyses et visualisations)

**Temps total estimé: ~5-7 minutes**

## 📊 Résultats Obtenus

### Fichiers Générés

1. **`data/annotations_balanced.csv`** - Dataset équilibré (10 images/classe)
2. **`results_evaluation_YYYYMMDD_HHMMSS.csv`** - Résultats détaillés avec prédictions

### Métriques Calculées

#### Métriques Globales
- ✅ Overall Accuracy
- ✅ Weighted F1-Score (tient compte du déséquilibre)
- ✅ Macro F1-Score (moyenne simple)
- ✅ Weighted Precision & Recall

#### Métriques par Classe
- ✅ Precision par classe
- ✅ Recall par classe
- ✅ F1-Score par classe
- ✅ Support (nombre d'échantillons)

#### Métriques de Latence
- ✅ Mean, Median, Std Dev
- ✅ P25, P50, P75, P95, P99
- ✅ Min, Max
- ✅ Throughput (req/s)

#### Analyse d'Erreurs
- ✅ Patterns de confusion les plus fréquents
- ✅ Heatmap des erreurs
- ✅ Liste détaillée des erreurs

## 📈 Avantages du Dataset Équilibré

### Pourquoi 10 images par classe?

**Problème avec le dataset original:**
- 1A: 25 images (16.7%)
- 2C: 4 images (2.7%)
- **Ratio de déséquilibre: 6.25:1**

**Solution avec le dataset équilibré:**
- Toutes les classes: 10 images chacune (10%)
- **Ratio de déséquilibre: 1:1** ✅

### Bénéfices

1. **Métriques Plus Fiables**
   - Chaque classe contribue également
   - Pas de biais vers les classes surreprésentées
   - F1-score macro devient pertinent

2. **Comparaison Équitable**
   - Même nombre d'échantillons pour chaque classe
   - Performance réelle du modèle visible
   - Facilite la comparaison entre APIs

3. **Temps d'Évaluation Raisonnable**
   - 100 images au lieu de 150
   - ~5 minutes au lieu de ~8 minutes
   - Résultats statistiquement significatifs

## 📝 Utilisation pour le Rapport

### Section 1: Dataset
Utilisez les cellules 3-4:
- Tableau de distribution original
- Graphiques de comparaison (original vs équilibré)
- Statistiques de déséquilibre

**À mentionner:**
> "Pour éviter les biais dus au déséquilibre des classes (ratio 6.25:1), nous avons créé un dataset équilibré avec 10 images par classe, permettant une évaluation plus juste des performances du modèle."

### Section 2: Métriques de Classification
Utilisez les cellules 7-9:
- Tableau des métriques globales
- Matrice de confusion
- Graphiques de performance par classe

**Screenshots à inclure:**
- Confusion matrix (absolue et normalisée)
- Graphiques de Precision/Recall/F1 par classe

### Section 3: Analyse de Latence
Utilisez les cellules 10:
- Statistiques de latence détaillées
- Graphiques de distribution
- Calculs de throughput

**Métriques clés:**
- Mean latency
- P95 latency (pour SLA)
- Throughput (req/s)

### Section 4: Analyse d'Erreurs
Utilisez la cellule 11:
- Patterns de confusion
- Heatmap des erreurs
- Interprétation

### Section 5: Résumé Exécutif
Utilisez la cellule 12:
- Rapport complet formaté
- Tous les chiffres clés
- Top/worst performing classes

## 🔄 Tester Plusieurs APIs

Pour comparer différentes APIs:

1. **Modifier `config/eval.yaml`:**
```yaml
api:
  provider: google_gemini  # ou anthropic_claude
  model: gemini-1.5-pro
```

2. **Créer un nouveau client** dans le notebook (cellule 6):
```python
# Pour Gemini (exemple)
from google.generativeai import GenerativeModel
client = GeminiClient(api_key=os.getenv("GEMINI_API_KEY"))
```

3. **Réexécuter le notebook**

4. **Comparer les résultats**:
   - Accuracy
   - F1-scores
   - Latence
   - Patterns d'erreurs

## 🐛 Dépannage

### Erreur: API Key non trouvée
```bash
# Vérifier le .env
cat .env

# Ou définir directement
export AZUREOPENAI_API_KEY="votre-clé"
```

### Erreur: Module non trouvé
```bash
pip install -r requirements.txt
```

### Erreur: Timeout
Augmenter le timeout dans `config/eval.yaml`:
```yaml
api:
  timeout: 60  # au lieu de 30
```

## ✅ Checklist Avant Exécution

- [ ] Environnement virtuel activé
- [ ] Variables d'environnement définies (.env chargé)
- [ ] Fichier `data/annotations.csv` présent
- [ ] Fichier `prompts/vision_only.txt` présent
- [ ] Fichier `config/eval.yaml` configuré
- [ ] Connexion internet stable

## 📊 Structure du Code

Toutes les fonctions utilisent les bonnes appellations:
- ✅ `EvalConfig` pour la configuration
- ✅ `AzureLLMClient` pour le client API
- ✅ `HairClassifier` pour le classificateur
- ✅ `Evaluator` pour l'évaluation
- ✅ Colonnes CSV: `image_path`, `type`
- ✅ Résultats: dictionnaire avec clés `accuracy`, `weighted_f1`, `confusion_matrix`, etc.

## 🎓 Code de Qualité

- Code commenté et documenté
- Structure logique en 12 étapes
- Gestion d'erreurs incluse
- Visualisations professionnelles
- Messages informatifs pour l'utilisateur

---

**Prêt à lancer?** Ouvrez `run.ipynb` et exécutez toutes les cellules! 🚀

**Questions?** Consultez [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) pour plus de détails.
