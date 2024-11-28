# SuperPoint with Illumination Conditions Adaptation (ICA)

This repository provides an inference pipeline for a PyTorch implementation of the SuperPoint network, enhanced with the **Illumination Conditions Adaptation (ICA)** method. 
Showcases the research and some of the methods developed in the following papers:

- **Increasing Illumination Invariance of Learning-Based Features through Realistic Simulated Environments Adaptation**  
  DOI: [-/-] (submitted for publication)

- **Illumination Conditions Adaptation for Data-Driven Keypoint Detection under Extreme Lighting Variations**  
  DOI: [10.1109/IST59124.2023.10355736](https://doi.org/10.1109/IST59124.2023.10355736)
  
  
  
 ![Demo Preview](assets/demo-results/output_ica.gif)
 ![Demo Preview](assets/demo-results/output_ica_2.gif)


## Installation

### Prerequisites
We tested the following 2 settings: 
- **Python** = 3.6 and 3.8
- **PyTorch** =1.7.1 and 2.1.2
- **OpenCV** =3.4.2 and 4.10
- **CUDA** 11.0 and 12.1

### Steps

1. Clone the repository:
   ```bash
    git clone https://github.com/anastaga/ica_sp.git
    cd ica_sp

2. Run the demo (use --cuda to run with GPU requires CUDA):
   ```bash
    python sp_inference.py assets/kitti06.mp4 --cuda   
    python sp_inference.py assets/night-kitti06 --cuda 

## Features

- **PyTorch Implementation of Superpoint:** 
  - Built upon existing SuperPoint implementations, based on the works of:
    - [Magicleap/SuperPointPretrainedNetwork](https://github.com/magicleap/SuperPointPretrainedNetwork) ([Original Paper](https://arxiv.org/abs/1712.07629))
    - [Eric-yyjau/pytorch-superpoint](https://github.com/eric-yyjau/pytorch-superpoint)
    - [Rpautrat/SuperPoint](https://github.com/rpautrat/SuperPoint)
    
- **Photo-Realistic Synthetic Illumination (PRSI) Dataset:**
  - Day-to-night dataset created using Unreal Engine 5, featuring diverse environments, controlled lighting transformations, and precise camera poses. Created to guide ICA's training.
 
<p align="center">
  <img src="/assets/img1_d.png" alt="" width="15%">
  <img src="/assets/img1_n.png" alt="" width="15%">
  <img src="/assets/img2_d.png" alt="" width="15%">
  <img src="/assets/img2_n.png" alt="" width="15%">
</p>



- **Illumination Conditions Adaptation (ICA):**
  - A novel technique leveraging day-time features as pseudo-ground truths for the training of night-time images.

- **Custom Night-KITTI Variants:**
  - Night-time datasets generated using [img2img-turbo](https://github.com/GaParmar/img2img-turbo), LUTs, and OpenCV, including:
    - `night-kitti00`
    - `night-kitti06`
    - `dark-night-kitti06`

- **Evaluation Pipelines:** 
  - Benchmarked using:
    - [HPatches Benchmark](https://github.com/hpatches/hpatches-dataset)
    - [PySLAM Toolkit](https://github.com/luigifreda/pyslam)

- **Demo Visualization Script:** 
  - Includes `sp_inference.py` for quick testing and visualization of feature detection results.



## Acknowledgements

 - [MagicLeap](https://github.com/magicleap/SuperPointPretrainedNetwork)




### Disclaimer
This repository does not include or distribute the original SuperPoint code. Users must independently acquire the SuperPoint implementation and weights under the license terms provided by Magic Leap, Inc. 

The work presented here focuses on novel contributions, including the PRSI dataset, Illumination Conditions Adaptation (ICA) method, and evaluation processes. All dependencies, including SuperPoint, and other referenced works, are properly credited, and users are advised to comply with their respective licenses.
