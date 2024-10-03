import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import models

num_labels = len(mlb.classes_)


model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_labels),  # num_labels is the number of possible labels
    nn.Sigmoid()  # Use sigmoid for multilabel classification
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming images and encoded_labels are ready
for epoch in range(num_epochs):
    for i, image_path in enumerate(images):
        image = load_image(image_path).unsqueeze(0)  # Add batch dimension
        label = torch.tensor(encoded_labels[i]).float().unsqueeze(0)  # Add batch dimension

        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()