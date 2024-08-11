import json
with open('fashion.json') as f:
    data = json.load(f)

with open('fashion-cat.json') as f:
    categories = json.load(f)

category_map = {cat['id']: cat['name'] for cat in categories}

images = [item['image_path'] for item in data]
labels = [[category_map[label_id] for label_id in item['label']] for item in data]