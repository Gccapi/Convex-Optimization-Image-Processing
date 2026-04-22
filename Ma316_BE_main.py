# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:23:43 2024

@author: 
    floal
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Ma316_BE_lib as m

#%% Partie 2 - Q11.d

P = np.array([[2, 1], [1, 2]])
Q = np.copy(P)

R = np.zeros_like(P)  # Matrice nulle de même taille que P
X0 = np.array([[1, 1],[2, 4]])

# Résolution de l'équation P*X + X*Q = R
X = m.grad_conj(m.A2, R, X0, 1e-6, 1000)



M=m.P@X+X@Q

#%% Partie 3.1 - Q1.2.c:Application numérique

#Chargement de la matrice et définition de sa taille
R = np.load('Data/R.npy')
M, N = np.shape(R)

#Valeurs très grandes donc le programme met du temps à s'effectuer 
#Mais le V trouvé finalement sera plus précis
itermax = 1000 #Plus rapide : V = 100
epsilon = 1e-6 #Plus rapide : epsilon=1e-4

#Initialisation des variables
e0 = 1
b = 1 / e0 * R
x0 = np.ones_like(R)

V = m.grad_conj(m.A3, b, x0, epsilon, itermax)

#%%  Partie 3 - Q1.2.d

M, N = np.shape(V) #V de la question précédente

DM = np.eye(M, dtype=float) * (0)
DM += np.eye(M, k=1, dtype=float)  * (-1/2)
DM += np.eye(M, k=-1, dtype=float) * (1/2)


DN = np.eye(N, dtype=float) * (0)
DN += np.eye(N, k=1, dtype=float)  * (-1/2)
DN += np.eye(N, k=-1, dtype=float) * (1/2)

#E_x = DM @ V
#E_y = V @ (DN.T)   fonction champ_elec les donne direct

delta_x = 1 
delta_y = 1
Ex, Ey = m.champ_elec(V, delta_x, delta_y)

# Calcul de E en utilisant les expressions 
E =np.sqrt(Ex**2 + Ey**2) 

# Création de la grille de coordonnées 
x=np.linspace(0,M-1,M)
y=np.linspace(0,N-1,N)

X, Y = np.meshgrid(y, x)

# Tracé des lignes de niveaux avec contourf
plt.figure(figsize=(12,8))
plt.contourf(R)

# Tracé des lignes de champ avec streamplot
plt.streamplot(X, Y, Ex, Ey, color='black', linewidth=0.7, arrowsize=1, density=1)

plt.contourf(V,levels=31,alpha=0.5)
plt.colorbar(label='Potentiel V') #échelle pour le potentiel

# Personnalisation du graphe
plt.xlabel('Potentiel')
plt.ylabel('Champ électrique E')
plt.title('Champ électrique en fonction du potentiel statique')
plt.gca().invert_yaxis() #inverse x et y 
plt.tight_layout() #pas utile mais plus propre

# Affichage du tracé
plt.show()


#%% Partie 3.1 Q3 

# Définition des valeurs de charge pour les charges positives et négatives
charge_positive = 1
charge_negative = -1

M, N = 512, 512  # Taille de la matrice de distribution des charges

# Récupérer les charges dessinées et l'image qui va avec 
R_dessin = m.dessin_charge(M, N)

# Vecteur initial
X0 = np.ones_like(R_dessin)

#fixation des limites et valeur initiale
epsilon = 1e-6
itermax = 1000
e0 = 1
b = 1 / e0 * R_dessin

# Application du gradient conjugué
V2 = m.grad_conj(m.A3, b, X0, epsilon, itermax)

# Calcul du champ électrique 
delta_x, delta_y = 1, 1
Ex, Ey = m.champ_elec(V2, delta_x, delta_y)

# Dimensions et grille pour l'affichage
xmax, ymax = M, N
x = np.linspace(0, xmax, M)
y = np.linspace(0, ymax, N)
X, Y = np.meshgrid(x, y)  

# Tracé des niveaux de potentiel électrostatique
plt.figure(figsize=(10, 8))

# Contourf pour les niveaux de potentiel
contour = plt.contourf(X, Y, V2, levels=31, cmap='coolwarm', alpha=0.8)
plt.colorbar(contour, label='Potentiel électrostatique (V)')

# Streamplot pour les lignes de champs électrique
plt.streamplot(X, Y, Ex, -Ey, color='black', linewidth=0.7, arrowsize=1, density=1)

# Mettre en évidence les charges dessinées
for y_idx in range(M):
    for x_idx in range(N):
        if R_dessin[y_idx, x_idx] > 0:  # Charge positive
            plt.plot(x_idx, y_idx, 'bo', markersize=8)  # Rouge pour positif (mais conflit avec les matrices et inversion entre BGR et RGB)
        elif R_dessin[y_idx, x_idx] < 0:  # Charge négative
            plt.plot(x_idx, y_idx, 'ro', markersize=8)  # Bleu pour négatif

# Configuration des axes et titres
plt.xlabel('x')
plt.ylabel('y')
plt.title('Champ électrique et potentiel électrostatique à partir des pôles fictifs')
plt.gca().invert_yaxis()  # Inverser l'axe y pour correspondre à la matrice
plt.tight_layout()

# Afficher le tracé
plt.show()


#%% Partie 3.2 Question 3: restauration

#Question 3a, b & c
F = cv2.imread("Data/orig1.jpg")
F = cv2.cvtColor(F, cv2.COLOR_BGR2RGB)  # Conversion en RGB
restored_img = m.img_restau_opti(F)

# Affiche l'image restaurée 
cv2.imshow('Image Restauree', cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Partie 3.2 - Question 4 : Ldiff

F = cv2.imread('Data/piscine.jpg')
G = cv2.imread('Data/ours.jpg')

# Fixer le paramètre alpha entre 0 et 1 
alpha = 0.9

# Résoudre le problème Idiff
img_modif = m.Ldiff(F, G, alpha)

# On affiche l'image modifiée
img_modif=np.maximum(0,np.minimum(img_modif,255)) #limite les valeurs pour rester dans la plage possible
img_modif = cv2.medianBlur(img_modif, 1) # on peut lisser l'image pour un résultat plus propre
cv2.imshow('Image modifiee', img_modif)
cv2.waitKey(0)
cv2.destroyAllWindows()