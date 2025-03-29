## AI Disclosure

### AI Assistance Overview
I used ChatGPT to assist in completing the basic code for Part 1, starting from the sample [`starter_code.py`](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/starter_code.py). Additionally, I consulted ChatGPT to identify the best non-transformer model for CIFAR-100 ([reference](https://chatgpt.com/c/67e19b18-4804-800d-ada9-a5020ca7fb2a)).

### AI Contribution vs. My Work
ChatGPT generated the basic structure of the code.
I selected and tuned hyperparameters, including learning rate, epochs, data augmentation strategies, optimizer, scheduler, and model, with guidance from ChatGPT.
For hyperparameter tuning, I provided ChatGPT with training and validation loss screenshots when I needed additional insights.
I also used ChatGPT to refine the wording in this report.

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
```

### Search Process & Reason
- Most were just trial and error results.
- Learning Rate Scheduler (ReduceLROnPlateau):
  - Factor (Amount of LR change): Initially set to 0.5 but later reduced to 0.1 for a more aggressive learning rate adjustment.
  - Patience (Epochs before adjustment): Experimented with values from 5 to 3 to make the learning rate decay more responsive.

---

## Regularization Techniques
- **Weight Decay:** `weight_decay=5e-4` in the Adam optimizer applies L2 regularization, which discourages large weights and helps prevent overfitting.

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

### Reason
- I duplicated the dataset and applied multiple levels of random noise transformations to maintain the original data while increasing robustness.
- After inspecting distorted images, I observed that most distortions were random noise rather than cropping or flipping. This insight guided my choice of augmentation techniques.
- **Reference:** You can check the [`vis.py`](https://github.com/dl4ds-gh-classroom/dl4ds-spring-2025-midterm-challenge-CaslowChien/blob/main/vis.py) [AI reference](https://chatgpt.com/share/67e76032-de0c-800d-8764-cd9c95657f2f) for visualization and further inspiration.

One kind of distortion
severity 1:
(add pic here)
severity 4:
(add pic here)

Another kind of distortion

---

## Results Analysis
- Discussion of model strengths, weaknesses, and areas for improvement.
- Comparison of training and validation performance.
- Error analysis and potential refinements.

### Model Performance
(add pic here)
Training loss decreased steadily, and validation loss showed minimal overfitting, indicating effective regularization.

### Strengths
+ Robustness to Noise: One of the ConvNeXt's architectural advantages is to handle noisy test samples well.
+ Efficient LR Scheduling: From my experience, ReduceLROnPlateau scheduler in our case ensured effective learning rate reductions when performance stagnated.
+ Augmentation Effectiveness: The augmentation strategy improved generalization by simulating real-world noise conditions.

### Areas for Improvement
+ Performance: The model stopped improving after epoch 30, showing space for improvement.
+ Hyperparameter Sensitivity: The model seemed sensitivity to learning rate adjustments.
---

## Experiment Tracking Summary
- Include screenshots or summaries from the experiment tracking tool.
- Utilize WandB Reports UI for a structured report presentation.

