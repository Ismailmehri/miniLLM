# MiniLLM Learning Project

Bienvenue dans le projet **MiniLLM**. Ce dépôt est conçu pour apprendre et expérimenter avec les Large Language Models (LLMs) à petite échelle.

## Objectifs

L'objectif est de maîtriser les concepts clés suivants :
- **Création de Mini LLMs** : Comprendre l'architecture et l'entraînement de modèles de langage.
- **Fine-tuning** : Adapter des modèles existants à des tâches spécifiques.
- **RAG (Retrieval-Augmented Generation)** : Augmenter les capacités des LLMs avec des connaissances externes.

## Structure du Projet

- `data/` : Datasets bruts et traités.
- `models/` : Checkpoints des modèles et modèles sauvegardés.
- `notebooks/` : Jupyter notebooks pour l'exploration et les tutoriels.
- `src/` : Code source du projet.
  - `data_loader/` : Scripts de chargement de données.
  - `training/` : Pipelines d'entraînement et de fine-tuning.
  - `rag/` : Implémentation des systèmes RAG.
  - `utils/` : Fonctions utilitaires.

## Installation

1. Cloner le dépôt :
   ```bash
   git clone <votre-repo-url>
   cd miniLLM
   ```

2. Créer un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

*Les instructions détaillées pour lancer les entraînements et les démos seront ajoutées ici.*
