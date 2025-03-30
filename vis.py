import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def visualize_images(distortion_name, severity, CONFIG, num_images=40):
    data_dir = CONFIG["ood_dir"]
    
    # Load the OOD images
    images = np.load(os.path.join(data_dir, f"{distortion_name}.npy"))

    # Select the subset of images for the given severity
    start_index = (severity - 1) * 10000
    end_index = severity * 10000
    images = images[start_index:end_index]

    # Convert to PyTorch tensors and normalize to [0, 1]
    images = torch.from_numpy(images).float() / 255.
    images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

    # Define the transformation
    transform = transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.0, 0, 1))
    

    # Apply the transformation
    transformed_images = torch.stack([transform(img) for img in images])

    # Show a grid of images
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate rows dynamically
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

    fig.suptitle(f"Sample Images - {distortion_name} (Severity {severity}) with Noise")
    for i, ax in enumerate(axes.flat):
        if i >= len(transformed_images): 
            ax.axis("off")
            continue
        img = transformed_images[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.axis("off")

    plt.show(block=True)

# Example usage
CONFIG = {"ood_dir": "./data/ood-test"}  # Change this to your actual path
# pad a zero if the number is one digit
# visualize_images("distortion00", 1, CONFIG)

visualize_images(f"distortion00", 1, CONFIG)
# for i in range(19):
#     if i < 10:
#         visualize_images(f"distortion0{str(i)}", 1, CONFIG)  # Visualize distortion06 10 times
#     else:
#         visualize_images(f"distortion{str(i)}", 1, CONFIG)  # Visualize distortion06 10 times
    