import numpy as np

def cost_reg_func(parameters, *args):
    
    # Obtient les arguments de la fonction
    Y, R, L, n_movies, n_users, n = args
    
    # "Déroule" le vecteur de paramètres
    parameters = parameters.reshape(n_movies + n_users, n)
    
    # "Sépare" X et theta
    X = parameters[:n_movies,:]
    theta = parameters[n_movies:,]
    
    # Calcul du coût réglularisé
    J = R * (np.dot(X, theta.T) - Y)
    J = 0.5 * (J**2)
    J = J.sum()
    J_reg = J + (L/2)*(X**2).sum() + (L/2)*(theta**2).sum()

    return J_reg


def grad_reg_func(parameters, *args):
    
    # Obtient les arguments de la fonction
    Y, R, L, n_movies, n_users, n = args
    
    # "Déroule" le vecteur de paramètres
    parameters = parameters.reshape(n_movies + n_users, n)
    
    # "Sépare" X et theta
    X = parameters[:n_movies,:]
    theta = parameters[n_movies:,]
    
    # Calcul du gradient
    X_grad = R*(np.dot(X, theta.T) - Y)
    X_grad = np.dot(X_grad, theta)
    
    theta_grad = R*(np.dot(X, theta.T) - Y)
    theta_grad = np.dot((theta_grad).T, X)
    
    X_grad_reg = X_grad + L * X;
    theta_grad_reg = theta_grad + L * theta;
    
    return np.vstack([X_grad_reg, theta_grad_reg]).flatten()