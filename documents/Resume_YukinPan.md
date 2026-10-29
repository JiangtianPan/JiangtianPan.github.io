Yukin Pan
======
###### [ [jiangtianpan.github.io](https://jiangtianpan.github.io) ] . [ jiangtianpan@hotmail.com ]


Education
---------
**The Ohio State University** (08/2017 - 05/2019)

- M.S. in Electrical and Computer Engineering.
- Coursework in Computer Vision, Machine Learning and Image Processing.

**Wuhan University** (09/2013 - 06/2017)
- B.S. in Automation, GPA 85/100 [Top 15%]
- AAA Graduate Student

Scholarships and Awards
---------
- 2018: AI Star Intern in the JD AI Research [10/50]
- 2018: Ranked 2nd in the CVPR 2018 Workshop, [LIP Contest](https://sysu-hcp.net/lip/parsing_lb.php?type=2)
- 2017: AAA Graduate Student [10/450]
- 2016: Interdisciplinary Contest in Modeling (Honorable Mention)
- 2015: WHU Outstanding Student Leader [2/60]
- 2014: Avery Dennison Foundation Spirit of Invention Scholarship [10/150]

Professional Experience
---------
**CV Engineer, vivo AI Research** (07/2019 - 06/2021)

> Face Clustering in Digital Album:
- **Facial Feature Extraction**: Built a CNN tailored to extracting human facial features.

- **Graph Clustering Algorithm (GCA)**: Designed a clustering algorithm grounded in graph structure and K-Nearest Neighbors (KNN) theory.

- **On-Device Implementation**: Paired GCA with cosine distance metrics for real-time facial clustering on smartphones.

- **Results**: Raised Accuracy/Recall from 0.92/0.79 without GCA to 0.99/0.94 with it.

- **Deployment and Scale**: Shipped the algorithmic module on over 1 million vivo smartphones.

> Real-Time Human Segmentation on Mobile Devices:
- **CNN Construction**: An Encoder-Decoder model built on [UNet](https://arxiv.org/pdf/1505.04597.pdf) and [Depthwise Separable Convolution](https://arxiv.org/pdf/1704.04861.pdf), implemented in TensorFlow.

- **Data Augmentation**: Grew the [PFCN](http://xiaoyongshen.me/webpage_portrait/index.html) dataset (18,698 images) and the vivo dataset (5,000 images) past 60,000 images via cropping, flipping, shifting, zooming and alpha blending.

- **Inter-Frame Consistency**: Introduced alpha blending during processing to keep masks stable frame to frame.

- **Computational Efficiency**: Distillation and pruning took inference from 25 fps to 75 fps on Qualcomm Snapdragon 855 platforms.

> Real-Time Frame Interpolation and Motion Deblur on Mobile Devices:
- **CNN Construction**: A two-stage model rooted in the [Super SloMo](https://arxiv.org/pdf/1712.00080.pdf) architecture, covering optical flow prediction and frame restoration.

- **Computational Efficiency**: Cut model FLOPs substantially through distillation and pruning.

- **Unsupervised Learning Strategy**: Proposed an unsupervised scheme enabling training on [Adobe240fps](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/) and [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) datasets.

- **Parallel Computing Pipeline Built**: Engineered a pipeline with OpenCL on GPU and VCap on NPU, taking throughput from 7 fps to 40 fps.

Research Experience
---------
**CV Researcher, Megvii Research** (07/2021 - present)

> Real-Time Video Quality Enhancement for Mobile Devices:
- **Proposed a CNN Based Model**: Combined an attention module, skip connections, residual modules and mask modules to restore video quality in real time.

- **Optimized Computational Efficiency**: From 330 GOPs to 12 GOPs by implementing **Space2Depth**, **Distillation** and **Pruning**. Achieved inference time reduction from 100+ ms to 15 ms on the Qualcomm Snapdragon 8475 platform.

- **Noise Data Collection**: Devised an approach to capture noise from real-world footage, improving training-set realism and model robustness.

- **Data Augmentation Techniques**: ISP simulation, brightness adjustment, motion simulation and random disturbance to widen the model's operating range.

- **Loss Function Design**: Formulated gradient loss, edge loss and mask loss to keep performance stable under difficult capture conditions.

- **Deployment**: Built a pipeline with pre-processing and post-processing on GPU and model inference on NPU. Integrated this algorithmic module into the XIAOMI 12 Pro smartphone.

> Real-World Blind Super-Resolution via Feature Matching:
- **Constructed a VQGAN Based Model**: Incorporated Residual Blocks, [Swin Transformer Blocks](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf), a Codebook and a Feature Matching Module to blindly enhance video resolution from 720p to 1440p.

- **HR Dataset Build**: Built and augmented a high-resolution (HR) dataset from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://arxiv.org/abs/1707.02921), [DIV8K](https://ieeexplore.ieee.org/document/9021973) and real-world datasets via texture filtering.

- **Data Enhancement**: Used pretrained models [FeMaSR](https://arxiv.org/abs/2202.13142), [Real-ESRGAN](https://arxiv.org/abs/2107.10833) and [BasicVSR++](https://ckkelvinchan.github.io/projects/BasicVSR++/) to further enhance high-frequency textures via image filtering and patch filtering.

- **Loss Function Design**: Formulated L2 loss, perceptual loss, GAN loss and codebook loss for robust model performance in the real world.

- **Model Pruning**: Pruned the model from 1600G FLOPs to 320G FLOPs (input shape 512x512x3) with almost the same performance.

**Research Intern, JD AI Research**, *Advisor: Prof. [Tao Mei](https://scholar.google.com.hk/citations?user=7Yq4wf4AAAAJ&hl=zh-CN)* (05/2018 - 09/2018)

> CVPR 2018 Workshop, LIP Contest [(Ranked 2nd)](https://sysu-hcp.net/lip/parsing_lb.php?type=2)
- **Data Augmentation**: Extended the training set with image rotations and reversals, plus Mask-RCNN for multi-human object detection.

- **Model Fine-Tuning and Fusion**: Customized and fine-tuned JPP-Net, SS-NAN, DenseNet, RefineNet and UPerNet on the LIP training set. Employed fusion strategies and integrated key-point data for performance refinement.

- **Hard Class Mining and Performance Optimization**: Identified and targeted several hard-to-classify categories, elevating overall model performance to 87.42% pixel accuracy.

Academic Publication
---------
- Philippe Zhang, et al. “*[Patch Progression Masked Autoencoder with Fusion CNN Network for Classifying Evolution Between Two Pairs of 2D OCT Slices](https://doi.org/10.1007/978-3-031-86651-7_2)*”, International Challenge on Device-Independent Diabetic Macular Edema Onset Prediction. Springer Nature Switzerland, 2024. 97-105.

- Philippe Zhang, et al. “*[Automatic Prediction of Center-Involved Diabetic Macular Edema Using Ultra-Widefield Color Fundus Photography](https://doi.org/10.1007/978-3-031-86651-7_9)*”, International Challenge on Device-Independent Diabetic Macular Edema Onset Prediction. Springer Nature Switzerland, 2024. 20-29.

- Yunhan Huang, Li Ding, Yun Feng, Jiangtian Pan. “*Epidemic Spreading in Random Walkers with Heterogeneous Interaction Radius*”, JSTAT 2016(10), 103501.

Skills
------
**Programming**: Python, Java, MATLAB, C++, OpenCL, CUDA

**Deep Learning Framework**: MegEngine, PyTorch, TensorFlow, TFLite, ONNX

**Computer Vision Tools**: OpenCV, Numpy, BasicSR

**Professional Fields**: Low-Level Vision, Image/Video Quality Enhancement, Real-Time CNN for Mobile Device
