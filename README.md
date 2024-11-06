# Pinned_Insect_CV

This repository is for the proceeding paper **Accelerating Pinned Specimen Digitization for Biodiversity Research: A Deep Learning Pipeline for Collaborative Robots**.

This repository provides resources for training a YOLO model to detect pinned insects in images. It includes the steps required to set up the environment, prepare the dataset, train and evaluate the model, and visualize the results.

## Workspace Setup

### 1. Platform Requirements

This experiment is designed to run on a platform with an **NVIDIA GPU**.

**Recommended Environment:**
- Python 3.10
- PyTorch 2.3.1+cu121

**Installation Resources:**
- [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
- [Previous PyTorch Versions](https://pytorch.org/get-started/previous-versions/)

### 2. Clone the Repository

Clone this repository and the annotation tool used in this project:

```bash
git clone https://github.com/FrankZhangIronly/Pinned_Insect_CV.git
git clone https://github.com/syangeng/ISAT_with_segment_anything.git
```

### 3. Download the Dataset

Download the insect image dataset from [Zenodo](https://zenodo.org/records/14046033?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjZhMzkzOTVjLTlmZDUtNDk1My1hZjlhLWY5MDI2ODY0NDU3NSIsImRhdGEiOnt9LCJyYW5kb20iOiIwZjlmY2JjMDJiZDVjMjc0NzI3MDE4YTA5MmJmMjE5MiJ9.n5B45kLrLe9EI1nXk2GWFiCm6S3AVcLgEIhYGFM-YYYzXamNabUp6FY2F1uxJCf52ag6TsQhg5Sss6gZgqX0dQ).

If you need to create your own dataset, please use the **ISAT tool** for annotation and export the YOLO version annotation files.
- ISAT Tool Instructions: [ISAT-SAM Documentation](https://isat-sam.documentation.link)

**Note:** Scripts for splitting data into train and test sets are available in `jupyter/create.ipynb`.

## Training and Validating the YOLO Model on Image Dataset

### 4. Run the Training Script

#### 4.1 Navigate to the Root Directory

Start by navigating to the root directory of the repository.

#### 4.2 Update Mosaic Augmentation (Optional)

To modify mosaic augmentation from the default value of 4 to 9, follow these steps:

- Open the file located at:
  ```
  YOUR_PYTHON_PATH/Python3.10/dist-packages/ultralytics/data/augment.py
  ```
- In the function `__init__`, change the parameter `n` from 4 to 9.

#### 4.3 Run the Training Command

Use the command below to start training:

```bash
# Display available parameter settings
python run.py train --help

# Run training with custom parameters
python run.py train --model_path <path_to_model.pt> --data <path_to_data.yaml> [other parameters]
```

### 5. Run the Validation Script

Use the command below to validate and test on datasets:

```bash
# Display available parameter settings
python run.py validate --help

# Run validation with custom parameters
python run.py validate --model_path <path_to_model.pt> --data <path_to_data1.yaml> <path_to_data2.yaml> ... --name <val_name1> <val_name2> ...
```

## Visualization of Training Process

### 6. View Training Progress

YOLO will generate training visualizations in the project folder. You can view the progress using the `plot_training` script in `jupyter/create.ipynb`.

## Testing on Videos and Label Extraction

### 7. Test on Videos

- Use the trained weights located at `best.pt` in your project folder.
- Test videos are stored in `dataset/videos/`.

In `jupyter/aug.ipynb`:
- Use the code blocks labeled **"Video1"** for inference on videos.
- The second and third code blocks handle label extraction.

---

## Recent Updates

### 1021 Update

The final version of the Jupyter Notebook file, `train_T4_mask.ipynb`, has been updated. It contains all the code for the experiment mentioned in the paper.

---
### 0610 update
current detection and segmentation result on test video:
<p align="center">
  <a href="other/vi720.mp4">
    <img src="other/vi.png" alt="Demo Video">
  </a>
</p>
---

---
### 0509 update
1. create utils folders for self-defined tools
2. build augmentation and visulization (boxes) tools
---

### Project Aims
Develop CV and ML models to aid in the digitization of over 25 million pinned insect specimens.

The specific scenario we aim to address is **assisting robotic arms in identifying specimens within drawers** and **locating the pins** to grasp the target.

<p align="center">
    <img height="300" src="other/img.png"/>
</p>

### Question Breakdown
1. Develop models for pinned insect identification. (Pre-training)
2. Develop video datasets for pinned insect detection in drawers.
3. Develop models for detection in drawers based on identification models. (Fine-tuning)
4. Integrate additional modules for segmentation of insect.

## Project Baseline
Models are evaluated on COCO val2017 dataset.

| Model         | mAP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(B) |
|---------------|----------------------|--------------------|-------------------|
| YOLOv8n       | 37.3                 | 3.2                | 8.7               |
| YOLOv8s       | 44.9                 | 11.2               | 28.6              |
| YOLOv8m       | 50.2                 | 25.9               | 78.9              |
| YOLOv8l       | 52.9                 | 43.7               | 165.2             |
| YOLOv8x       | 53.9                 | 68.2               | 257.8             |
| RT-DETR-R18   | 46.5                 | 20                 | 60                |
| RT-DETR-R34   | 48.9                 | 31                 | 92                |
| RT-DETR-R50-m | 51.3                 | 36                 | 100               |
| RT-DETR-R50   | 53.1                 | 42                 | 136               |
| RT-DETR-R101  | 54.3                 | 76                 | 259               |
