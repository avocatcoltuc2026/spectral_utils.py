import numpy as np

def aliniere_spectrala_procruste(X_openAI, Y_cohere):
    """
    Aliniază spațiul latent OpenAI la cel Cohere păstrând distanțele geodezice interne.
    
    Parametri:
    ----------
    X_openAI : numpy.ndarray
        Matrice de formă (N, d1) reprezentând entitățile în spațiul OpenAI.
    Y_cohere : numpy.ndarray
        Matrice de formă (N, d2) reprezentând aceleași entități în spațiul Cohere.
        
    Returnează:
    -----------
    X_aliniat : numpy.ndarray
        Matricea OpenAI proiectată și rotită izometric în sistemul de coordonate Cohere.
    """
    # 1. Centrarea matricilor pentru eliminarea componentelor de translație
    X_centered = X_openAI - np.mean(X_openAI, axis=0)
    Y_centered = Y_cohere - np.mean(Y_cohere, axis=0)
    
    # 2. Calculul matricei de corelație încrucișată (Cross-Covariance)
    M = np.dot(X_centered.T, Y_centered)
    
    # 3. Descompunerea în Valori Singulare (SVD) pentru extragerea rotației optime
    U, S, Vt = np.linalg.svd(M)
    
    # 4. Calculul matricei de rotație ortogonală R
    R = np.dot(U, Vt)
    
    # 5. Proiecția spațiului OpenAI în noul sistem invariant
    X_aliniat = np.dot(X_centered, R)
    
    return X_aliniat
