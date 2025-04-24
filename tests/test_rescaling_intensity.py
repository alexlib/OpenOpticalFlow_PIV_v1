import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the implementation
from openopticalflow.rescaling_intensity import rescaling_intensity

def test_rescaling_intensity():
    """Test the rescaling_intensity function"""

    # Create test cases
    test_cases = []

    # Test case 1: Random images with different ranges
    np.random.seed(42)
    img1 = np.random.rand(100, 100) * 100  # Range [0, 100]
    img2 = np.random.rand(100, 100) * 200  # Range [0, 200]
    test_cases.append(("Random images", img1, img2))

    # Test case 2: Constant images
    img1 = np.ones((50, 50)) * 10
    img2 = np.ones((50, 50)) * 20
    test_cases.append(("Constant images", img1, img2))

    # Test case 3: Images with negative values
    img1 = np.random.rand(50, 50) * 200 - 100  # Range [-100, 100]
    img2 = np.random.rand(50, 50) * 100 - 50   # Range [-50, 50]
    test_cases.append(("Images with negative values", img1, img2))

    # Test case 4: Images with different shapes
    img1 = np.random.rand(40, 60) * 100
    img2 = np.random.rand(40, 60) * 150
    test_cases.append(("Different shapes", img1, img2))

    # Set max intensity value
    max_intensity = 255

    # Run tests
    for name, img1, img2 in test_cases:
        print(f"\nTesting {name}:")

        # Apply the implementation
        img1_rescaled, img2_rescaled = rescaling_intensity(img1.copy(), img2.copy(), max_intensity)

        # Check output types
        print(f"Output types:")
        print(f"  Rescaled: {img1_rescaled.dtype}, {img2_rescaled.dtype}")

        # Check output ranges
        print(f"Output ranges:")
        print(f"  Rescaled: [{img1_rescaled.min():.2f}, {img1_rescaled.max():.2f}], [{img2_rescaled.min():.2f}, {img2_rescaled.max():.2f}]")

        # Plot results
        plt.figure(figsize=(12, 8))

        # Original images
        plt.subplot(221)
        plt.imshow(img1, cmap='gray')
        plt.colorbar()
        plt.title(f'Original Image 1\nRange: [{img1.min():.2f}, {img1.max():.2f}]')

        plt.subplot(222)
        plt.imshow(img2, cmap='gray')
        plt.colorbar()
        plt.title(f'Original Image 2\nRange: [{img2.min():.2f}, {img2.max():.2f}]')

        # Rescaled images
        plt.subplot(223)
        plt.imshow(img1_rescaled, cmap='gray')
        plt.colorbar()
        plt.title(f'Rescaled Image 1\nRange: [{img1_rescaled.min():.2f}, {img1_rescaled.max():.2f}]')

        plt.subplot(224)
        plt.imshow(img2_rescaled, cmap='gray')
        plt.colorbar()
        plt.title(f'Rescaled Image 2\nRange: [{img2_rescaled.min():.2f}, {img2_rescaled.max():.2f}]')

        plt.tight_layout()
        plt.savefig(f'results/rescaling_intensity_{name.replace(" ", "_").lower()}.png')
        plt.close()

        # Verify that the rescaled images have the expected properties
        assert img1_rescaled.min() >= 0, f"Minimum value of rescaled image 1 should be >= 0, got {img1_rescaled.min()}"
        assert img2_rescaled.min() >= 0, f"Minimum value of rescaled image 2 should be >= 0, got {img2_rescaled.min()}"
        assert img1_rescaled.max() <= max_intensity, f"Maximum value of rescaled image 1 should be <= {max_intensity}, got {img1_rescaled.max()}"
        assert img2_rescaled.max() <= max_intensity, f"Maximum value of rescaled image 2 should be <= {max_intensity}, got {img2_rescaled.max()}"

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Run tests
    test_rescaling_intensity()
