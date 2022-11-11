## Code for Network Administration course, using Deep Neural Network to classify DDoS Attack's type in NSL-KDD Dataset

Use Anaconda or Miniconda to create environment for run project with command:

```
conda env create -f environment.yml
```

### Accuracy and loss with VGG:

|           Accuracy            |              Loss              |
|:-----------------------------:|:------------------------------:|
| ![](./image/VGG_acc_plot.png) | ![](./image/VGG_loss_plot.png) |

Inference time : ~32ms

### Accuracy and loss with DenseNet:

|              Accuracy              |                Loss                 |
|:----------------------------------:|:-----------------------------------:|
| ![](./image/DenseNet_acc_plot.png) | ![](./image/DenseNet_loss_plot.png) |

Inference time : ~29ms
