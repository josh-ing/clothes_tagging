from sklearn.metrics import precision_score, recall_score, f1_score
import torch

class model_evalulator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, images, labels):
        y_true = []
        y_pred = []

        for i, image in enumerate(images):
            image = image.unsqueeze(0)
            label = torch.tensor(labels[i]).float().unsqueeze(0)

            with torch.no_grad():
                output = self.model(image)
                preds = (output > 0.5).float()

            y_true.append(label.squeeze().cpu().numpy())
            y_pred.append(preds.squeeze().cpu().numpy())

        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')

        print(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")
