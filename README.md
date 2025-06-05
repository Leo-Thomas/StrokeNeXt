<div align="center">
  
# StrokeNeXt: A Siamese-encoder Approach for Brain Stroke Classification in Computed Tomography Imagery
<!--[Leo Thomas Ramos](https://www.linkedin.com/in/leo-thomas-ramos/)|, [Henry Valesaca](https://ec.linkedin.com/in/henry-velesaca-lara/)|, [Ángel D. Sappa](https://es.linkedin.com/in/angel-sappa-61532b17) -->
</div>

<!-- This repository is the official Pytorch implementation for [SkyScenes](). -->

<!-- [![Website](https://img.shields.io/badge/Project-Website-orange)](https://hoffman-group.github.io/SkyScenes/) [![arXiv](https://img.shields.io/badge/arXiv-SkyScenes-b31b1b.svg)](#)  -->


<!-- [![Watch the Demo](./assets/robust_aerial_videos.mp4)](./assets/robust_aerial_videos.mp4) -->

<img src="./assets/teaser.png" width="100%"/>

## Announcements

StrokeNeXt is under review <!--at []() ! 📣 -->

## About the project

We present StrokeNeXt, a deep learning model designed for stroke classification in 2D Computed Tomography (CT) imagery. The architecture integrates two identical ConvNeXt encoders in a dual-branch configuration, enabling complementary feature extraction from the same input. The outputs are merged via a lightweight convolutional decoder composed of stacked 1D operations, including a bottleneck projection and transformation layers, followed by a compact classifier that produces the final prediction.

### Paper

[#](#)

## Dataset download

We use the Stroke Dataset released as part of the Artificial Intelligence in Healthcare Competition (TEKNOFEST 2021). It consists of 6,774 brain CT cross-sectional images in PNG format, annotated and curated by a team of seven radiologists. The dataset is divided into three classes: 4,551 non-stroke, 1,093 hemorrhagic stroke, and 1,130 ischemic stroke cases. The dataset is free available [here](https://www.kaggle.com/datasets/orvile/inme-veri-seti-stroke-dataset).

To structure the dataset for our experiments, we reorganized the original set into two distinct classification scenarios. The first focuses on detecting the presence of stroke, grouping ischemic and hemorrhagic cases into a single class and using the non-stroke cases as the negative class. The second scenario targets stroke subtype classification, using only the ischemic and hemorrhagic samples and excluding non-stroke cases. For each setup, the data was randomly split into training, validation, and test sets following an 80-10-10 proportion.

## Results

### Performance classification of stroke presence (non-stroke or stroke)

| Method | Accuracy | Precision | Recall | F1-score | Training time |  Inference time |
|------|------|------|------|------|------|------|
|MobileNetv2 | 0.868 | 0.866 | 0.868 | 0.865 | 0.040 | 0.0002| 
|VGG16 | 0.894 | 0.893 | 0.894 | 0.893 | 0.041 | 0.0002|
|ResNet50 | 0.856 | 0.854 | 0.856 | 0.853 | 0.040 | 0.0002| 
|ResNet152 | 0.846 | 0.843 | 0.846 | 0.843 | 0.046 | 0.0003|
|Swin Transformer | 0.893 | 0.893 | 0.892 | 0.890 | 0.067 | 0.0004| 
|ConvNeXt-base | 0.875 | 0.875 | 0.875 | 0.874 | 0.049 | 0.0002|
|StrokeNeXt-large | 0.987 | 0.987 | 0.987 | 0.987 | 0.245 | 0.0003|

### Performance classification of stroke subtypes (ischemia or hemorrhage)

| Method | Accuracy | Precision | Recall | F1-score | Training time |  Inference time |
|------|------|------|------|------|------|------|
|MobileNetv2 | 0.843 | 0.845 | 0.843 | 0.842 | 0.020 | 0.0001|
|VGG16 | 0.892 | 0.893 | 0.892 | 0.892 | 0.020 | 0.0002|
|ResNet50 | 0.807 | 0.811 | 0.807 | 0.806 | 0.022 | 0.0002|
|ResNet152 | 0.791 | 0.791 | 0.792 | 0.791 | 0.025 | 0.0003|
|Swin Transformer | 0.796 | 0.795 | 0.795 | 0.795 | 0.029 | 0.0003|
|ConvNeXt-base | 0.852 | 0.854 | 0.852 | 0.852 | 0.026 | 0.0002|
|StrokeNeXt-base | 0.988 | 0.988 | 0.988 | 0.988 | 0.101 | 0.0003|

### Checkpoints (non-stroke or stroke)

| Encoder | Accuracy | Precision | Recall | F1-score | Training time |  Inference time |  Checkpoint |
|------|------|------|------|------|------|------|------|
|ConvNeXt-tiny | 0.978 | 0.978 | 0.978 | 0.978 | 0.087 | 0.0002|
|ConvNeXt-small | 0.980 | 0.981 | 0.981 | 0.980 | 0.126 | 0.0003|
|ConvNeXt-base | 0.982 | 0.983 | 0.982 | 0.982 | 0.161 | 0.0003|
|ConvNeXt-large | 0.987 | 0.987 | 0.987 | 0.987 | 0.245 | 0.0003|

### Checkpoints (ischemia or hemorrhage)

| Encoder | Accuracy | Precision | Recall | F1-score | Training time |  Inference time |  Checkpoint |
|------|------|------|------|------|------|------|------|
|ConvNeXt-tiny | 0.986 | 0.985 | 0.986 | 0.986 | 0.041 | 0.0002|
|ConvNeXt-small | 0.987 | 0.987 | 0.987 | 0.987 | 0.096 | 0.0004| 
|ConvNeXt-base | 0.988 | 0.988 | 0.988 | 0.988 | 0.101 | 0.0005| 
|ConvNeXt-large | 0.987 | 0.987 | 0.985 | 0.986 | 0.122 | 0.0006| 

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

## License

Distributed under GNU General Public License v3.0. See `LICENSE` for more information.

## BibTex

If you find this dataset useful, please star ⭐️⭐️⭐️ our repo and cite our paper.

```
soon
```

