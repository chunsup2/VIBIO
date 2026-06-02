# VIB-IO
Pytorch implementation of "Approximating the Ideal Observer via a Variational Information Bottleneck", submitted to the journal of Medical Imaging special issue honoring Harrison H. Barrett.

## Abstract
Purpose: The Bayesian ideal observer (IO) defines an upper bound on task-based observer performance and is widely used as a theoretical benchmark for optimizing medical imaging systems and data-acquisition designs. However, except in special cases, the exact IO test statistic is analytically intractable.
Recent deep neural network (DNN)-based approaches show promise for approximating IO performance, but they provide limited analytical insight and cannot guarantee that the learned test statistic corresponds to a likelihood ratio. In addition, their empirical performance can degrade in data-limited regimes, motivating methods that incorporate stronger statistical structure and regularization.
To address this, we propose a data-driven framework based on the variational information bottleneck (VIB) to approximate the IO test statistic for binary signal-detection tasks.

Approach: The proposed method utilizes the variational bottleneck principle to map high-dimensional image data into a low-dimensional latent space optimized to preserve task-relevant information while suppressing nuisance variability. An analytically motivated Gaussian IO loss encourages the class-conditional latent distributions to follow multivariate Gaussian models. Under this Gaussian latent-space representation, the likelihood ratio can be computed analytically, combining the representational capacity of DNNs with the tractability of model-based observers.

Resutls: Numerical studies demonstrate that the proposed VIB-based IO method achieves detection performance comparable to conventional convolutional neural network (CNN) approaches on large datasets, and superior performance in data-limited regimes. Furthermore, the learned bottleneck representations exhibit  stability under the domain shifts examined, supporting robust deployment across varying imaging conditions. 

Conclusions: The proposed VIB-IO framework provides a theoretically grounded and relatively data-efficient alternative to conventional CNN-based IO approximations, while retaining greater analytical transparency through an explicit latent-space Gaussian observer model.
By combining the representational capacity of DNNs with an analytically tractable latent-space observer model, the proposed framework provides a principled approach for objective, task-based assessment of medical imaging systems.


## Notice
This repository is currently a **work in progress**. Only the core parts of the initial version of the code have been uploaded. If you have any questions, please feel free to open an issue so we can make this repo better.




