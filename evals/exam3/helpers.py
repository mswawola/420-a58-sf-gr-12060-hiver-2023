import numpy as np


def content_based_cost_reg_func(parameters, *args):
    
    # Obtient les arguments de la fonction
    X, Y, R, L, n_movies, n_users, n = args
    
    # "Déroule" le vecteur de paramètres
    theta = parameters.reshape(n_users, n)
    
    # Compléter le code ci-dessous ~ 1-4 lignes de code
    
    J = R * (np.dot(X, theta.T) - Y)
    J = 0.5 * (J**2)
    J = J.sum()
    J_reg = J + (L/2)*(X**2).sum() + (L/2)*(theta**2).sum()

    return J_reg


def content_based_grad_reg_func(parameters, *args):
    # Obtient les arguments de la fonction
    X, Y, R, L, n_movies, n_users, n = args
    
    # "Déroule" le vecteur de paramètres
    theta = parameters.reshape(n_users, n)
    
    # Compléter le code ci-dessous ~ 6 lignes de code
    
    theta_grad = R*(np.dot(X, theta.T) - Y)
    theta_grad = np.dot((theta_grad).T, X)
    
    theta_grad_reg = theta_grad + L * theta;
    
    # Cette fonction doit retourner les gradients sous forme d'un seul vecteur
    return theta_grad_reg.flatten()