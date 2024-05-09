# Pinned_Insect_CV
This repository is for the UCL MSc EDS dissertation project **Computer Vision and Machine Learning for Pinned Insect Identification**. 

---
## 0509 update
1. create utils folders for self-defined tools
2. build augmentation and visulization (boxes) tools
---

## Project Aims
Develop CV and ML models to aid in the digitization of over 25 million pinned insect specimens.

The specific scenario we aim to address is **assisting robotic arms in identifying specimens within drawers** and **locating the correct part** to grasp the target.

<p align="center">
    <img height="300" src="other/img.png"/>
</p>

## Question Breakdown
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
