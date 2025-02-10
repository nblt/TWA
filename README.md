# TWA
The code is the official implementation of our paper 
[Trainable Weight Averaging: Accelerating Training and Improving Generalization].

Weight averaging is a widely used technique for accelerating training and improving the generalization of deep neural networks (DNNs). While existing approaches like stochastic weight averaging (SWA) rely on pre-set weighting schemes, they can be suboptimal when handling diverse weights. We introduce Trainable Weight Averaging (TWA), a novel optimization method that operates within a reduced subspace spanned by candidate weights and learns optimal weighting coefficients through optimization. TWA offers greater flexibility and can be applied to different training scenarios. For large-scale applications, we develop a distributed training framework that combines parallel computation with low-bit compression for the projection matrix, effectively managing memory and computational demands. TWA can be implemented using either training data (TWA-t) or validation data (TWA-v), with the latter providing more effective averaging. Extensive experiments showcase TWA's advantages: (i) it consistently outperforms SWA in generalization performance and flexibility, (ii) when applied during early training, it reduces training time by over 40\% on CIFAR datasets and 30\% on ImageNet while maintaining comparable performance, and (iii) during fine-tuning, it significantly enhances generalization by weighted averaging of model checkpoints. In summary, we present an efficient and effective framework for trainable weight averaging. 


## Dependencies

Install required dependencies:

```[bash]
pip install -r requirements.txt
```

## How to run

### Training from scratch
```[bash]
bash run_cifar.sh
```


### Fine-tuning

```[bash]
bash twa_v.sh
```




## Citation
If you find this work helpful, please cite:
```
@inproceedings{
    li2023trainable,
    title={Trainable Weight Averaging: Efficient Training by Optimizing Historical Solutions},
    author={Tao Li and Zhehao Huang and Qinghua Tao and Yingwen Wu and Xiaolin Huang},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=8wbnpOJY-f}
}
```