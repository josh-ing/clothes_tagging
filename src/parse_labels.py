import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder

class parse_labels:
    def __init__(self, fashion_file, category_file):
        self.fashion_file = fashion_file
        self.category_file = category_file
        self.images = []
        self.labels = []
        self.encoded_labels = []
        self.category_map = {}

    def load_data(self):
        with open(self.fashion_file) as f:
            data = json.load(f)

        with open(self.category_file) as f:
            categories = json.load(f)

        self.category_map = categories
        # self.images = [item['image_path'] for item in data]
        for item in data:
            product_id = item['product']
            scene_id = item['scene']
            bbox = item['bbox']

            category = self.category_map.get(product_id)

            if category:
                self.images.append(scene_id)
                self.labels.append(category)
        # self.labels = [[self.category_map[label_id] for label_id in item['label']] for item in data]

    def encode_labels(self):
        encoder = LabelEncoder()
        self.encoded_labels = encoder.fit_transform(self.labels)
        return encoder.classes_

    def get_data(self):
        return self.images, self.encoded_labels