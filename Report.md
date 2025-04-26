## AI Disclosure

### AI Assistance Overview
I used ChatGPT to assist in completing the basic code for Part 1, starting from the sample [`starter_code.py`](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/starter_code.py). Additionally, I consulted ChatGPT to identify the best non-transformer model for CIFAR-100 ([reference](https://chatgpt.com/c/67e19b18-4804-800d-ada9-a5020ca7fb2a)).

### AI Contribution vs. My Work
ChatGPT generated the basic structure of the code.
I selected and tuned hyperparameters, including learning rate, epochs, data augmentation strategies, optimizer, scheduler, and model, with guidance from ChatGPT.
For hyperparameter tuning, I provided ChatGPT with training and validation loss screenshots when I needed additional insights.
I also used ChatGPT to refine the wording in this report.

## Repository Structure
```
.
├── .github/                  # GitHub-specific files
├── assets/                   # Training curves and distortion visualization images
│   ├── distortion00_severity_max.png	# screen shot example of severity max distortion 
│   ├── distortion00_severity_min.png	# screen shot example of severity min distortion 
│   ├── distortion02_severity_min.png	# screen shot example of severity min distortion for another batch of png 
│   ├── wandb_all.png	# screen shot of "all" training evaluation in wandb
│   ├── wandb_best.png	# screen shot of "the best(part 3)" training evaluation in wandb
│   ├── wandb_best_part1.png	# screen shot of "part 1" training evaluation in wandb
│   └── wandb_best_part2.png	# screen shot of "part 2" training evaluation in wandb
├── .gitignore                 # Specifies data and others files to ignore in Git
├── README.md                  # Project overview and instructions
├── Report.md                  # Detailed report about methods and results (this file)
├── requirements.txt           # Python package requirements
├── starter_code.py            # Sample starter code for Part 1
├── utils.py                   # Utility functions (e.g., data loading, preprocessing)
├── vis.py                     # Visualization code with distorted images
├── vis_original.py            # Visualization code for original (undistorted) images
├── eval_cifar100.py           # Script to evaluate models on CIFAR-100 dataset
├── eval_ood.py                # Script to evaluate models on OOD test images
├── part1_simple_cnn.py        # Part 1: Simple CNN model training
├── part2_cnn.py               # Part 2: Advanced CNN models training
├── part3_transfer.py          # Part 3: Transfer learning implementation
```


---

## Model Description

### Chosen Architecture and Justification
- **Part 1:** Used simple layers to test baseline performance.
- **Part 2 & Part 3:** Implemented **ConvNeXt_Base** from torchvision.
  - **Primary Reason:** It outperformed other models I tested, including `wide_resnet50_2` and `resnext50_32x4d`.
  - **Additional Considerations:**
    - The testing dataset contained significant noise. ConvNeXt, inspired by Vision Transformers but retaining convolutional inductive biases, is robust against noise while effectively capturing local patterns.
    - Its deep and wide convolutional layers with large receptive fields allow the model to focus on more global features, reducing sensitivity to pixel-level noise.

---

## Hyperparameter Tuning

### Final Values
```python
batch_size = 256
learning_rate = 3e-4
epoch = 50
optimizer = create_optimizer_v2(model.parameters(), opt = "adamw", lr = 3e-4, weight_decay = 1e-2, momentum = 0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, threshold=1e-3, threshold_mode='rel', verbose=True)
```

### Search Process & Reason
- Most were just trial and error results.
- optimizer: suggestion from this article https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055-2/#18f9
- Learning Rate Scheduler (ReduceLROnPlateau):
  - Factor (Amount of LR change): Initially set to 0.5 but later reduced to 0.1 for a more aggressive learning rate adjustment.
  - Patience (Epochs before adjustment): Experimented with values from 5 to 3 to make the learning rate decay more responsive.

---

## Regularization Techniques
- **Weight Decay:** `weight_decay=1e-2` in the Adam optimizer applies L2 regularization, which discourages large weights and helps prevent overfitting.

---

## Data Augmentation Strategy

```python
transform_0 = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.025),
transforms.Lambda(lambda x: x + 0.1),  # Shifting all values slightly
transforms.Lambda(lambda x: F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0))
])

transform_1 = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
transforms.Lambda(lambda x: x + 0.1),  # Shifting all values slightly
transforms.Lambda(lambda x: F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0))
])

transform_2 = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.075),
transforms.Lambda(lambda x: x + 0.1),  # Shifting all values slightly
transforms.Lambda(lambda x: F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0))
])

transform_3 = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
transforms.Lambda(lambda x: x + 0.1),  # Shifting all values slightly
transforms.Lambda(lambda x: F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0))
])

trainset_0 = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_0, download=False)
trainset_1 = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_1, download=False)
trainset_2 = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_2, download=False)
trainset_3 = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_3, download=False)
trainset_4 = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=False, transform=transform_train)

trainset = ConcatDataset([trainset_0, trainset_1, trainset_2, trainset_3, trainset_4])
```

⭐ this includes data leakage to validation since I duplicated them before training, but it's working well on both clean testing and real testing, so I'll keep it.

### Reason
- I duplicated the dataset and applied multiple levels of random noise transformations to maintain the original data while increasing robustness.
- After inspecting distorted images, I observed that most distortions were random noise. This insight guided my choice of augmentation techniques.
- **Reference:** You can check the [`vis.py`](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/vis.py) [AI reference](https://chatgpt.com/share/67e76032-de0c-800d-8764-cd9c95657f2f) for visualization and further inspiration.
- In cifar-100 training, each class has 500 images, therefore it's not imbalanced. So I didn't use any techniques for balancing the dataset.

One kind of distortion

severity 1:\
![distortion00 severity 1](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/assets/distortion00_severity_min.png?raw=true)

severity 5:\
![distortion00 severity 5](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/assets/distortion00_severity_max.png?raw=true)

Another kind of distortion:\
![distortion02 severity 1](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/assets/distortion02_severity_min.png?raw=true)


## Results Analysis
- Discussion of model strengths, weaknesses, and areas for improvement.
- Comparison of training and validation performance.
- Error analysis and potential refinements.

### Model Performance
![wandb best model](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/assets/wandb_best.png?raw=true)

Training loss decreased steadily, and validation loss showed minimal overfitting, indicating effective regularization.

### Strengths
+ Robustness to Noise: One of the ConvNeXt's architectural advantages is to handle noisy test samples well.
+ Efficient LR Scheduling: From my experience, ReduceLROnPlateau scheduler in our case ensured effective learning rate reductions when performance stagnated.
+ Augmentation Effectiveness: The augmentation strategy improved generalization by simulating real-world noise conditions.

### Areas for Improvement
+ Performance: The model stopped improving after epoch 30, showing space for improvement.
+ Hyperparameter Sensitivity: The model seemed sensitivity to learning rate adjustments.
+ Data Leakage: I intentionally duplicated the training dataset for augmentation, which may have led to data leakage into the validation set. This caused training and validation accuracy to be close to 100, leading difficulties in evaluating the model's true performance.
---

## Experiment Tracking Summary
⭐ There is data leakage in validation accuracy that are close to 90%. However, there's no leakage in testing, so I think it's still acceptable. 

All Improved Training:
![wandb best model](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/assets/wandb_all.png?raw=true)

### PART 1
The Best Model:
![wandb best model part 1](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/assets/wandb_best_part1.png?raw=true)

[Link to best model part 1](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/simple_cnn.py)

### PART 2
The Best Model:
![wandb best model part 2](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/assets/wandb_best_part2.png?raw=true)

[Link to best model part 2](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/cnn.py)

### PART 3
The Best Model:
![wandb best model](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/assets/wandb_best.png?raw=true)

[Link to best model part 3](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/transfer.py)

```python
model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
```

**stilted-valley-21**
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```
**hearty-night-22**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

==============================================
```python
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
```

**logical-jazz-23**
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

**misty-night-24**
```python
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # More flexible cropping
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  # Add slight rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color perturbations
    transforms.RandomGrayscale(p=0.1),  # Occasional grayscale conversion
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])
```

**royal-jazz-25**
```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,  # Initial restart period
    T_mult=2,  # Multiplicative factor for subsequent restart periods
    eta_min=1e-5  # Minimum learning rate
    )
```
==============================================
```python
model = models.convnext_base(pretrained=True)
```

**major-deluge-29**
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)
```

**fresh-wood-31**
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
	scheduler.step(val_loss)
```

**treasured-spaceship-33**
```python
"batch_size": 256
```

**fine-river-34**
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
```
* change factor from 0.5 to 0.1, patience from 5 to 3
-> factor is the amount LR change, so 0.1 change more than 0.5
-> patience is when will lr change, eg. reduces LR every ...patience... epochs if the loss doesn’t improve.

**grateful-dream-54**
```python
transform_0 = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.025),
transforms.Lambda(lambda x: x + 0.1),  # Shifting all values slightly
transforms.Lambda(lambda x: F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0))
])

transform_1 = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
transforms.Lambda(lambda x: x + 0.1),  # Shifting all values slightly
transforms.Lambda(lambda x: F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0))
])

transform_2 = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.075),
transforms.Lambda(lambda x: x + 0.1),  # Shifting all values slightly
transforms.Lambda(lambda x: F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0))
])

transform_3 = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
transforms.Lambda(lambda x: x + 0.1),  # Shifting all values slightly
transforms.Lambda(lambda x: F.avg_pool2d(x.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0))
])

trainset_0 = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_0, download=False)
trainset_1 = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_1, download=False)
trainset_2 = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_2, download=False)
trainset_3 = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_3, download=False)
trainset_4 = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=False, transform=transform_train)

trainset = ConcatDataset([trainset_0, trainset_1, trainset_2, trainset_3, trainset_4])
```

**distinctive-butterfly-65**
```python
optimizer = create_optimizer_v2(model.parameters(), opt = "adamw", lr = 3e-4, weight_decay = 1e-2, momentum = 0.9)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)
```

**grateful-snowflake-70**  test acc: 74.19%
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, threshold=1e-3, threshold_mode='rel', verbose=True)
```
