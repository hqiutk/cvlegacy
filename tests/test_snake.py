import numpy as np
import matplotlib.pyplot as plt

import cvlegacy.snake_segmentation as snake_segmentation

def create_synthetic_image(size=200, circle_center=None, circle_radius=50):
    """
    Create a synthetic image with a bright circular object on a dark background

    Args:
        size (int, optional): image size (size x size). Defaults to 200.
        circle_center (_type_, optional): center of the circle. Defaults to None - uses the image center.
        circle_radius (int, optional): radius of the circle. Defaults to 50.
        
    Returns:
        image: 2D numpy array (grayscale)
    """
    if circle_center is None:
        circle_center = (size // 2, size // 2)
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    image = np.zeros((size, size))
    mask = (X - circle_center[0])**2 + (Y - circle_center[1])**2 <= circle_radius**2
    image[mask] = 1.0
    return image

def main():
    # create a synthetic image containing a circular object
    image = create_synthetic_image(size=200, circle_radius=50)
    
    # define the initial snake (a circle that encloses the object)
    init_snake = create_initial_snake(num_points=100, center=(100, 100), radius=70)
    
    # run the discrete snake segmentation algorithm
    final_snake = snake_segmentation(image, init_snake, alpha=0.2, beta=0.2, gamma=1.0,
                                     iterations=200, window_size=1)
    
    # plot the initial snake and the final snake over the synthetic image
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(image, cmap='gray')
    ax[0].plot(init_snake[:, 1], init_snake[:, 0], 'r-', lw=2)
    ax[0].set_title('Initial Snake')
    ax[0].axis('off')
    
    ax[1].imshow(image, cmap='gray')
    ax[1].plot(final_snake[:, 1], final_snake[:, 0], 'r-', lw=2)
    ax[1].set_title('Final Snake Segmentation')
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()
     
    