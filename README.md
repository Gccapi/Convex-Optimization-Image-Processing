# Optimisation Convexe & Traitement d'Image

Ce projet met en œuvre des méthodes d'optimisation numérique pour résoudre des problèmes physiques et de traitement d'image complexe.
Le but est d'utiliser cette optimisation mathématique pour réaliser des montages photo, tels que l'incrustation de visages sur des statues, d'éléments d'une image au sein d'une autre.

## Le défi : Fusionner sans limite nette
Le problème majeur d'un copier-coller classique est la différence de luminosité et de couleur entre les images. Ce projet résout ce problème en utilisant l'**Équation de Poisson**. 
Au lieu de copier les pixels, on copie le gradient et on laisse l'algorithme recalculer les couleurs pour qu'elles s'adaptent à la nouvelle image.

## Concepts clés
- **Gradient Conjugué** : Implémentation d'un solveur itératif pour les systèmes linéaires de grande taille (Equation $Ax = b$).
- **Électrostatique** : Résolution de l'équation de Poisson pour modéliser le potentiel électrique entre des charges.
- **Édition d'Image (Poisson Blending)** : Fusion invisible d'images (Seamless Cloning) en minimisant l'énergie de gradient entre une source et une cible.



## Utilisation
Le projet utilise **OpenCV** pour la manipulation matricielle des images et **tqdm** pour le suivi de la convergence de l'algorithme.
- Lancer `python Ma316_BE_main.py` pour voir les démonstrations de restauration et de fusion d'images.

## Auteurs
- Florian Alaux
- Elina Noll
- Ranihei Peirsegaele
- Laurie Torregrossa
