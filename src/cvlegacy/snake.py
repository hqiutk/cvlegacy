import numpy as np
import matplotlib.pylot as plt

def snake_segmentation(image, init_snake, alpha=0.2, beta=0.2, gamma=1.0, iterations=200, window_size=1):
    """
    Discrete snake segmentation algorithm.

    Args:
        image (2D numpy array): grayscale input image
        init_snake [(N,2) array]: initial snake coordinates (row, col)
        alpha (float, optional): weight of the internal energy (1st derivative). Defaults to 0.2.
        beta (float, optional): weight of the internal energy (2nd derivative). Defaults to 0.2.
        gamma (float, optional): weight of the external energy. Defaults to 1.0.
        iter (int, optional): number of iterations to run. Defaults to 200.
        window_size (int, optional): search window radius (in pixels) for local optimization. Defaults to 1.
    
    Returns:
        snake: (N,2) array of final snake coordinates.
    """
    # find the external energy
    gy, gx = np.gradient(image.astype(float))
    edge = -np.exp(gx**2 + gy**2)
    
    # copy the initial snake
    snake = init_snake.copy()
    N = snake.shape[0]
    
    # iterate to update each snake point
    for iter in range(iterations):
        new_snake = snake.copy()
        for i in range(N):
            # for each snake point, search in a window of radius=window_size
            for dx in range(-window_size, window_size+1):
                for dy in range(-window_size, window_size+1):
                    candidate = snake[i] + np.array([dx, dy])
                    
                    # internal energy
                    prev_point = snake[(i-1) % N]
                    next_point = snake[(i-1) % N]
                    
                    E_int1 = np.sum((candidate - prev_point)**2)
                    E_int2 = np.sum((next_point + prev_point - 2 * candidate)**2)
                    
                    # external energy
                    x_c, y_c = int(round(candidate[0])), int(round(candidate[1]))
                    if x_c < 0 or x_c >= image.shape[0] or y_c < 0 or y_c >= image.shape[1]:
                        E_ext = 1e5 
                    else:
                        E_ext = edge[x_c, y_c]
                        
                    E_total = alpha * E_int1 + beta * E_int2 + gamma * E_ext
                    if E_total < best_energy:
                        best_energy = E_total
                        best_position = candidate
            new_snake[i] = best_position
        snake = new_snake
    return snake

def create_initial_snake(num_points=100, center=(100, 100), radius=70):
    """
    Create an initial snake as a circle.

    Args:
        num_points (int, optional): number of points along the snake. Defaults to 100.
        center (tuple, optional): center of the circle. Defaults to (100, 100).
        radius (int, optional): radius of the initial snake circle. Defaults to 70.
        
    Returns:
        snake: (num_points,2) array of snake coordinates (row, col).
    """
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False) 
    snake = np.zeros((num_points, 2))
    # use sine for row coordinates and cosin for column coordinates.
    snake[:, 0] = center[0] + radius * np.sin(theta)
    snake[:, 1] = center[1] + radius * np.cos(theta)
    return snake
