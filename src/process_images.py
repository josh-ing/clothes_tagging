from PIL import Image
from torchvision import transforms

class process_images:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, signature):
        prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
        print(prefix % (signature[0:2], signature[2:4], signature[4:6], signature))
        return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)
    