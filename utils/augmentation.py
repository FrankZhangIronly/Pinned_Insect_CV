import os
from PIL import Image
import random
from tqdm import tqdm


class aug_crop:

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.input_image_dir = os.path.join(base_dir, 'images')
        self.input_label_dir = os.path.join(base_dir, 'labels')
        self.output_image_dir = os.path.join(base_dir, 'images_crop')
        self.output_label_dir = os.path.join(base_dir, 'labels_crop')

        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_label_dir, exist_ok=True)

    def _load_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = f.readlines()
        return [list(map(float, label.strip().split())) for label in labels]

    def _save_labels(self, label_file, labels):
        with open(label_file, 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')

    def _calculate_global_bbox_selfpad(self, labels, img_width, img_height, padding_min=0.1, padding_max=0.3):
        x_mins = []
        y_mins = []
        x_maxs = []
        y_maxs = []

        padding = random.uniform(padding_min, padding_max)

        for label in labels:
            class_id, cx, cy, w, h = label
            x_min = (cx - w / 2) * img_width
            y_min = (cy - h / 2) * img_height
            x_max = (cx + w / 2) * img_width
            y_max = (cy + h / 2) * img_height

            x_mins.append(x_min)
            y_mins.append(y_min)
            x_maxs.append(x_max)
            y_maxs.append(y_max)

        x_min_global = int(min(x_mins))
        y_min_global = int(min(y_mins))
        x_max_global = int(max(x_maxs))
        y_max_global = int(max(y_maxs))

        box_width, box_height = x_max_global - x_min_global, y_max_global - y_min_global

        x_min_global = max(0, int(min(x_mins) - padding * box_width))
        y_min_global = max(0, int(min(y_mins) - padding * box_height))
        x_max_global = min(img_width, int(max(x_maxs) + padding * box_width))
        y_max_global = min(img_height, int(max(y_maxs) + padding * box_height))

        return x_min_global, y_min_global, x_max_global, y_max_global

    def _calculate_global_bbox(self, labels, img_width, img_height, padding_min=0.05, padding_max=0.1):
        x_mins = []
        y_mins = []
        x_maxs = []
        y_maxs = []

        padding = random.uniform(padding_min, padding_max)

        for label in labels:
            class_id, cx, cy, w, h = label
            x_min = (cx - w / 2) * img_width
            y_min = (cy - h / 2) * img_height
            x_max = (cx + w / 2) * img_width
            y_max = (cy + h / 2) * img_height

            x_mins.append(x_min)
            y_mins.append(y_min)
            x_maxs.append(x_max)
            y_maxs.append(y_max)

        x_min_global = max(0, int(min(x_mins) - padding * img_width))
        y_min_global = max(0, int(min(y_mins) - padding * img_height))
        x_max_global = min(img_width, int(max(x_maxs) + padding * img_width))
        y_max_global = min(img_height, int(max(y_maxs) + padding * img_height))

        return x_min_global, y_min_global, x_max_global, y_max_global

    def _update_labels(self, labels, x_min_global, y_min_global, new_img_width, new_img_height, img_width, img_height):
        new_labels = []
        for label in labels:
            class_id, cx, cy, w, h = label
            class_id = int(class_id)
            cx_new = (cx * img_width - x_min_global) / new_img_width
            cy_new = (cy * img_height - y_min_global) / new_img_height
            w_new = w * img_width / new_img_width
            h_new = h * img_height / new_img_height
            new_labels.append([class_id, cx_new, cy_new, w_new, h_new])
        return new_labels

    def crop(self):
        image_files = [f for f in os.listdir(self.input_image_dir) if f.endswith('.jpeg')]

        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(self.input_image_dir, img_file)
            label_path = os.path.join(self.input_label_dir, img_file.replace('.jpeg', '.txt'))

            img = Image.open(img_path)
            img_width, img_height = img.size

            labels = self._load_labels(label_path)

            x_min_global, y_min_global, x_max_global, y_max_global = self._calculate_global_bbox(labels, img_width,
                                                                                                 img_height)

            cropped_img = img.crop((x_min_global, y_min_global, x_max_global, y_max_global))
            new_img_width, new_img_height = cropped_img.size

            new_labels = self._update_labels(labels, x_min_global, y_min_global, new_img_width, new_img_height,
                                             img_width,
                                             img_height)

            cropped_img_path = os.path.join(self.output_image_dir, img_file)
            cropped_img.save(cropped_img_path)

            cropped_label_path = os.path.join(self.output_label_dir, img_file.replace('.jpeg', '.txt'))
            self._save_labels(cropped_label_path, new_labels)
