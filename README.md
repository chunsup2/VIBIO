# VIB-IO
Pytorch implementation of "Approximating the Ideal Observer via a Variational Information Bottleneck", submitted to the journal of Medical Imaging special issue honoring Harrison H. Barrett.

## Abstract
Dual-energy computed tomography (CT) is an excellent substitute for identifying bone marrow edema in magnetic resonance imaging. However, it is rarely used in practice owing to its low contrast. To overcome this problem, we constructed a framework based on deep learning techniques to screen for diseases using axial bone images and to identify the local positions of bone lesions. To address the limited availability of labeled samples, we developed a new generative adversarial network (GAN) that extends expressions beyond conventional augmentation (CA) methods based on geometric transformations. We theoretically and experimentally determined that combining the concepts of data augmentation optimized for GAN training (DAG) and Wasserstein GAN yields a considerably stable generation of synthetic images and effectively aligns their distribution with that of real images, thereby achieving a high degree of similarity. The classification model was trained using real and synthetic samples. Consequently, the GAN technique used in the diagnostic test had an improved F1 score of approximately 7.8% compared with CA. The final F1 score was 80.24%, and the recall and precision were 84.3% and 88.7%, respectively. The results obtained using the augmented samples outperformed those obtained using pure real samples without augmentation. In addition, we adopted explainable AI techniques that leverage a class activation map (CAM) and principal component analysis to facilitate visual analysis of the network’s results. The framework was designed to suggest an attention map and scattering plot to visually explain the disease predictions of the network


## Notice
This repository is currently a **work in progress**. Only the core parts of the initial version of the code have been uploaded. If you have any questions, please feel free to open an issue so we can make this repo better.




