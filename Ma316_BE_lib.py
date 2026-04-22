# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:25:32 2024

@author: 
    floal 
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

#%% Partie 2 - Q10

# Définition de la fonction de produit scalaire
def ps(x, y):
    return np.trace(x.T @ y) 


# Définition de la première application linéaire A(x)
def A2(x):
   global P,Q
   P=np.array([[2, 1], [1, 2]])
   Q=np.copy(P)
   return P @ x + x @ Q
 
''' premier test fonction gradient conjugué:
    
def grad_conj(ps, App, b, x0, epsilon, itermax):
    # Initialisation des variables
    x = np.copy(x0)
    d = b - App(x)  # Utilisation de la fonction A(x) avec les matrices P et Q
    compteur = 0   
    y = x + epsilon * np.ones(np.shape(x))
    
    # Initialisation de tqdm (pour suivre la progression sur les excecussions longues)
    with tqdm(total=itermax, desc="Calcul de grad_conj") as pbar:
        # Boucle jusqu'au rang k = itermax ou convergence
        while np.linalg.norm(x - y) > epsilon and compteur < itermax:
            y = np.copy(x)
            t = -(ps(d, App(x) - b)) / (ps(d, App(d)))
            x = x + t * d  # Incrémentation au rang k+1
            beta = (ps(d, App(App(x) - b) - App(x))) / (ps(d, App(d)))
            d = -(App(x) - b) + beta * d
            compteur += 1
            
            # Mise à jour de la barre de progression
            pbar.update(1)
        
    return np.maximum(0,np.minimum(x,255))

'''       
# Définition de la fonction de gradient conjugué
def grad_conj(A, b, x0, epsilon, itermax):
    # On commence avec une version float (évite les pb avec les images entières)
    x = x0.astype(np.float64)
    r = b - A(x)  # Résidu de base
    p = r.copy()  # Direction de descente initiale
    rs_old = np.sum(r * r)  # Norme au carré du résidu

    # Initialisation de tqdm (pour suivre la progression sur les itérations longues pour être sur que le code tourne)
    with tqdm(total=itermax, desc="Calcul de grad_conj") as pbar:
        for _ in range(itermax):
            Ap = A(p)  # Applique A sur la direction
            alpha = rs_old / np.sum(p * Ap)  # Optimisation dans la direction de descente
            x += alpha * p  # Mise à jour de la solution
            r -= alpha * Ap  # Mise à jour du résidu
            rs_new = np.sum(r * r)
            if np.sqrt(rs_new) < epsilon:  # Si le résidu est petit, on arrête
                break
            p = r + (rs_new / rs_old) * p  # Mise à jour de la direction
            rs_old = rs_new
            
            # Mise à jour de la barre de progression
            pbar.update(1)

    return x

#%%ParTie 3.1 - Q1.2.c

def A3(x): #Définition de l'application Linéaire de la question
    delta_x = 1
    delta_y = 1
    M, N = np.shape(x)
        
    LM = np.eye(M, dtype=int) * (-2) + np.eye(M, k=1, dtype=int) + np.eye(M, k=-1, dtype=int)
    LN = np.eye(N, dtype=int) * (-2) + np.eye(N, k=1, dtype=int) + np.eye(N, k=-1, dtype=int)
    
    x_LM = LM @ x  # Applique LM sur chaque colonne de x
    x_LN = x @ LN  # Et LN sur chaque ligne de x

    # Calcul de A
    A = (1 / delta_x**2) * x_LM + (1 / delta_y**2) * x_LN
        #Comme delta_x et delta_y = 1, on a : A = LM@x + x@LN
    #On laisse au cas où comme le paramètre peut être changé par l'utilisateur
        
    return A

#definition du champs électrique
def champ_elec(V, delta_x, delta_y):
    Ex = -np.gradient(V, delta_x, axis=1)
    Ey = -np.gradient(V, delta_y, axis=0)
    return Ex, Ey

#%% Partie 3.1 q3


def dessin_charge(M, N):
    matrice_charge = np.zeros((M, N))

    def cercle(event, x, y, flags, param):
        nonlocal matrice_charge
        if event == cv2.EVENT_LBUTTONDOWN or flags == cv2.EVENT_FLAG_LBUTTON:
            matrice_charge[y, x] = -1  # Ajout d'une charge positive
            cv2.circle(charge_image, (x, y), 5, (0, 0, 255), -1)  # Dessin d'un cercle rouge
        elif event == cv2.EVENT_RBUTTONDOWN or flags == cv2.EVENT_FLAG_RBUTTON:
            matrice_charge[y, x] = 1  # Ajout d'une charge négative
            cv2.circle(charge_image, (x, y), 5, (255, 0, 0), -1)  # Dessin d'un cercle bleu

    charge_image = np.zeros((M, N, 3), dtype=np.uint8)
    cv2.namedWindow('Charge Distribution')
    cv2.setMouseCallback('Charge Distribution', cercle)

    while True:
        cv2.imshow('Charge Distribution', charge_image)
        key = cv2.waitKey(1) & 0xFF
        if cv2.waitKey(20) & 0xFF == 27:  # echap pour sortir
            break

    cv2.destroyAllWindows()
    return matrice_charge

#%% Partie 3.2 : graffiti

# - Q2.3.a

# Fonction pour dessiner un masque sur une image
def draw_mask(F, pointe=5):
    # Variables globales pour suivre la position et l'état de dessin
    global ix, iy, drawing, img, M
    ix, iy = -1, -1  # Position initiale de la souris (inutilisée au départ)
    drawing = False  # On commence sans dessiner
    img = F.copy()  # Image qu'on modifie directement en affichage
    M = np.zeros(img.shape[:2], dtype=np.uint8)  # Masque (même taille que l'image, mais mono-canal)

    # Cette fonction s'exécute à chaque interaction souris
    def draw(event, x, y, flags, param):
        global ix, iy, drawing, img
        if event == cv2.EVENT_LBUTTONDOWN:  # Début du dessin
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Quand on bouge en dessinant
            # On trace une ligne sur l'image et le masque
            cv2.line(img, (ix, iy), (x, y), (0, 145, 255), pointe)  # en orange sympa
            cv2.line(M, (ix, iy), (x, y), 1, pointe)  # Trace la même chose sur le masque (en blanc)
            ix, iy = x, y
        elif event == cv2.EVENT_LBUTTONUP:  # Fin du dessin
            drawing = False

    # On configure une fenêtre pour afficher l'image et capter les clics souris
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw)

    # Boucle pour afficher l'image et quitter avec Échap
    while True:
        cv2.imshow('Image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Affiche en couleurs normales
        if cv2.waitKey(20) & 0xFF == 27:  # echap pour sortir
            break

    cv2.destroyAllWindows()
    return M  # Retourne le masque dessiné


# Fonction principale pour restaurer une image
def img_restau_opti(F, pointe=5, epsilon=1e-3, itermax=200, margin=4):
    mask = draw_mask(F, pointe)  # On récupère le masque dessiné à la souris
    Fc = F.astype(float) / 255  # Normalisation (plus pratique pour les calculs)
    M, N, p = Fc.shape  # Dimensions de l'image

    # On trouve les bords du masque et on rajoute une petite marge
    coords = np.argwhere(mask == 1)  # Coordonnées des pixels du masque
    min_x, max_x = np.min(coords[:, 0]) - margin, np.max(coords[:, 0]) + margin
    min_y, max_y = np.min(coords[:, 1]) - margin, np.max(coords[:, 1]) + margin
    # On s'assure de ne pas dépasser les limites de l'image
    min_x, max_x = max(0, min_x), min(M, max_x)
    min_y, max_y = max(0, min_y), min(N, max_y)

    # Découpe de l'image et du masque autour de la zone à corriger
    Mreduc = mask[min_x:max_x, min_y:max_y]
    Freduc = Fc[min_x:max_x, min_y:max_y, :]
    Ureduc = np.zeros_like(Freduc, dtype=np.float64)  # Résultat initial (vide)

    # Définition des opérateurs de Laplace
    def laplace(U):
        # On applique un opérateur 2D de Laplace discret
        return (
            -4 * U
            + np.roll(U, 1, axis=0) + np.roll(U, -1, axis=0)
            + np.roll(U, 1, axis=1) + np.roll(U, -1, axis=1)
        )

    def A(U):
        return laplace(U) * Mreduc  # Applique Laplace

    # On applique l'algorithme de gradient conjugué pour chaque canal de couleur
    for i in range(p):
        b = -laplace(Freduc[:, :, i] * (1 - Mreduc)) * Mreduc  # Calcul du résidu
        x0 = np.zeros_like(Mreduc, dtype=np.float64)  # Point de départ
        Ureduc[:, :, i] = grad_conj(A, b, x0, epsilon, itermax)

    # On réinsère la zone restaurée dans l'image complète
    U = Fc.copy()
    U[min_x:max_x, min_y:max_y, :] = Ureduc
    U = np.clip(U, 0, 1) * 255  # Re-normalisation
    U = U.astype(np.uint8)

    # Fusion de l'image originale avec la restaurée, selon le masque
    for i in range(p):
        U[:, :, i] = F[:, :, i] * (1 - mask) + U[:, :, i] * mask

    return U


#%% Partie 3.2: ours et piscine

def draw_poly(F):
    
    global points, M, img
    img = F.copy()
    points = [] # lise des points selectionnés pour dessiner le poly
    M = np.zeros(img.shape) #Matrice du masque initialisée à zéro et à la même taille que l'image
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points.clear()  # Effacer les anciens points
            points.append((x, y))  #ajouter le premier point
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                points.append((x, y))  #Ajoute les points lors du déplacement de la souris
                cv2.line(img, points[-2], points[-1], (0, 145, 255), 3)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            points.append((x, y))  # Ajoute le dernier point
            cv2.fillPoly(img, [np.array(points)], (0, 145, 255)) #Remplir le poly dessiné avec de la couleur
            cv2.fillPoly(M, [np.array(points)],(1, 1, 1), 1)  # Remplir le masque avec des pixels blancs
            M[np.array(points)[:, 1], np.array(points)[:, 0]] = 1 #Les points appartenant aux polygonnes ont des valeurs de 1

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    while True:
        cv2.imshow('Image', img)
        #cv2.imshow('Masque', M)
        if cv2.waitKey(1) & 0xFF == 27:  # Quitter si le echap pressé
            break

    cv2.destroyAllWindows()

    return M

#résolution de Ldiff optimisée

def Ldiff(F, G, alpha):    #gère l'intégration de l'image G dans F
    global Mreduc, LN, LM
    # Sélection d'une zone à modifier sur l'image F
    maskF = draw_poly(F)[:,:,0]
    F = F.astype(float)
    M,N,p = np.shape(F)
    # Sélection zone à modifier sur G
    maskG = draw_poly(G)[:,:,0]
    G = G.astype(float)
    
    #Coordonnées du masque de F
    CF = np.argwhere(maskF==1)
    min_l = np.min(CF[:, 0])
    max_l = np.max(CF[:, 0])
    min_c = np.min(CF[:, 1])
    max_c = np.max(CF[:, 1])
    
    # Les sous_matrices du masque et de l'image F + marge pour éviter les problèmes au bord
    Mreduc = maskF[min_l-3:max_l+4, min_c-3:max_c+4]
    Freduc = F[min_l-3:max_l+4, min_c-3:max_c+4,:]
    Ureduc = np.zeros(np.shape(Freduc))
    m,n = np.shape(Mreduc)
    
    # pareil pour G
    CG = np.argwhere(maskG==1)
    min_lG = np.min(CG[:, 0])
    max_lG = np.max(CG[:, 0])
    min_cG = np.min(CG[:, 1])
    max_cG = np.max(CG[:, 1])
    
    Greduc = G[min_lG-3:max_lG+4, min_cG-3:max_cG+4,:]
    Greduc = cv2.resize(Greduc,(n,m))
          
    global LM, LN        # Les matrices de Laplace
    LM = -(np.diag(-2*np.ones(m))+np.diag(np.ones(m-1), -1)+np.diag(np.ones(m-1), 1))
    LN = -(np.diag(-2*np.ones(n))+np.diag(np.ones(n-1), -1)+np.diag(np.ones(n-1), 1))
    
    def Adif(U):
        global LN, LM, Mreduc
        return (LM@U + U@LN)*Mreduc
    
    # On applique le gradient conjugué pour chaque canal de couleur
    for i in range(p):
        b = -Adif(Freduc[:,:,i] * (1 - Mreduc)) + alpha * Adif(Greduc[:,:,i]) + (1 - alpha) * Adif(Freduc[:,:,i])  
        x0 = np.ones_like(Mreduc)
        Ureduc[:,:,i] = grad_conj(Adif, b, x0, epsilon=1e-1, itermax=5000)
        Ureduc[:,:,i] = np.clip(Ureduc[:,:,i], 0, 255)  # limite les valeurs
   
    U = np.zeros(np.shape(F))
    U[min_l-3:max_l+4, min_c-3:max_c+4,:] = Ureduc
    
    for i in range(p):
        U[:,:,i] = F[:,:,i]*(1-maskF) + U[:,:,i]
    
    return U.astype(np.uint8)