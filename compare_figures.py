"""
compare_figures.py

This script compares the Python-generated figures with the MATLAB figures.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def compare_figures(matlab_dir='matlab_figures', python_dir='python_figures'):
    """
    Compare the Python-generated figures with the MATLAB figures.
    
    Args:
        matlab_dir (str): Directory containing MATLAB figures
        python_dir (str): Directory containing Python figures
    """
    # Create output directory for comparison figures
    output_dir = 'comparison_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of Python figures
    python_figures = sorted([f for f in os.listdir(python_dir) if f.endswith('.jpg')])
    
    # Get list of MATLAB figures
    matlab_figures = sorted([f for f in os.listdir(matlab_dir) if f.endswith('.jpg')])
    
    print(f"Found {len(python_figures)} Python figures and {len(matlab_figures)} MATLAB figures")
    
    # Compare figures
    for i, py_fig in enumerate(python_figures):
        if i < len(matlab_figures):
            mat_fig = matlab_figures[i]
            
            # Load images
            py_img = mpimg.imread(os.path.join(python_dir, py_fig))
            mat_img = mpimg.imread(os.path.join(matlab_dir, mat_fig))
            
            # Create comparison figure
            plt.figure(figsize=(15, 8))
            
            plt.subplot(121)
            plt.imshow(mat_img)
            plt.title(f'MATLAB: {mat_fig}')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(py_img)
            plt.title(f'Python: {py_fig}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'comparison_{i+1}.jpg'), dpi=150)
            plt.close()
            
            print(f"Compared {mat_fig} with {py_fig}")
    
    print(f"All comparison figures saved to {output_dir}/")

if __name__ == "__main__":
    compare_figures()
