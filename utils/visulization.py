from ultralytics import utils
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm

class vis_boxes:
    def __init__(self, base_dir):
        self.color_list = \
            [
                (255, 0, 0),
                (0, 255, 0),
                (0, 255, 255),
                (0, 128, 255),
                (127, 0, 255),
                (0, 0, 255),
            ]

        self.base_dir = base_dir
        self.output_label_dir = os.path.join(base_dir, 'labels_crop')

    def _vis_image_boxes(self, path, target, image_format = "jpeg"):
        label_path = Path(path)
        image_path = label_path.parents[1] / target / (path.split("\\")[-1][:-3]+image_format)
        im = Image.open(image_path)
        im_pic = utils.plotting.Annotator(im)
        for line in open(label_path).readlines():
            label_id, cx,cy,w,h = line.strip().split(" ")
            cx,cy,w,h = float(cx),float(cy),float(w),float(h)
            box=[(cx-w/2)*im.width,(cy-h/2)*im.height,(cx+w/2)*im.width,(cy+h/2)*im.height]
            im_pic.box_label(box=box, label=label_id, color=self.color_list[int(label_id)])
        im_pic.save(str(image_path).replace(target, target+"_box"))

    def show_image_boxes(self, path, target, image_format = "jpeg"):
        label_path = Path(path)
        image_path = label_path.parents[1] / target / (path.split("\\")[-1][:-3]+image_format)
        im = Image.open(image_path)
        im_pic = utils.plotting.Annotator(im)
        for line in open(label_path).readlines():
            label_id, cx,cy,w,h = line.strip().split(" ")
            cx,cy,w,h = float(cx),float(cy),float(w),float(h)
            box=[(cx-w/2)*im.width,(cy-h/2)*im.height,(cx+w/2)*im.width,(cy+h/2)*im.height]
            im_pic.box_label(box=box, label=label_id, color=self.color_list[int(label_id)])
        im_pic.show()

    def save_images_boxes(self, target):
        labeled_images_dir = os.path.join(self.base_dir, target+'_box')

        os.makedirs(labeled_images_dir, exist_ok=True)

        label_files = [f for f in os.listdir(self.output_label_dir) if f.endswith('.txt')]

        for label_file in tqdm(label_files, desc="Drawing boxes"):
            path = os.path.join(self.output_label_dir, label_file)
            self._vis_image_boxes(path, target)