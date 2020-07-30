# Quantization-aware-Training 
This module is about the 8-bit Quantization-aware-Training (QAT) for both forward and backward propagation stages.  
The demo codes are based on AlexNet with CIFAR-10 dataset.  
QAT compresses the model parameters and gradients for FP and BP, respectively.

### Recommended Development Environment
* PyTorch-1.4.0
* CUDA-10.1 (if GPU is enabled)

### Alexnet_qat  Folder Structure:  

* `Alex_qat.py`     : main entry of QAT training on alexnet
 
* `Timer.py`        : information log and time tools used in training procudre

* `config.py`       : detailed configuration of this project  
 
* `quantizerAlex.py`: quantizer API  
                
### Demo Results (AlexNet+CIFAR10):   

**In 10 Epochs: the model converges well without accuracy degradation**
![avatar](./demo_results/QAT_BP_Alex_perepoch_10log_for_10_epoch.png)

**In 50 Epochs: with 86% top-1 accuracy**
![avatar](./demo_results/QAT_BP_Alex_perepoch_1log_for_50_epoch.png)