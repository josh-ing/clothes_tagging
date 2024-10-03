import torch
import torch.optim as optim
import torch.nn as nn

class model_trainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, images, labels, num_epochs):
        for epoch in range(num_epochs):
            for i, image in enumerate(images):
                image = image.unsqueeze(0)  # Add batch dimension
                label = torch.tensor(labels[i]).float().unsqueeze(0)  # Add batch dimension

                self.optimizer.zero_grad()
                outputs = self.model(image)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
