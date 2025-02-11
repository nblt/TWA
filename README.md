# TWA
The code is the official implementation of our paper 
[Trainable Weight Averaging: Accelerating Training and Improving Generalization](http://arxiv.org/abs/2205.13104).

Weight averaging is a widely used technique for accelerating training and improving the generalization of deep neural networks (DNNs). While existing approaches like stochastic weight averaging (SWA) rely on pre-set weighting schemes, they can be suboptimal when handling diverse weights. We introduce Trainable Weight Averaging (TWA), a novel optimization method that operates within a reduced subspace spanned by candidate weights and learns optimal weighting coefficients through optimization. TWA offers greater flexibility and can be applied to different training scenarios. For large-scale applications, we develop a distributed training framework that combines parallel computation with low-bit compression for the projection matrix, effectively managing memory and computational demands. TWA can be implemented using either training data (TWA-t) or validation data (TWA-v), with the latter providing more effective averaging. Extensive experiments showcase TWA's advantages: (i) it consistently outperforms SWA in generalization performance and flexibility, (ii) when applied during early training, it reduces training time by over 40\% on CIFAR datasets and 30\% on ImageNet while maintaining comparable performance, and (iii) during fine-tuning, it significantly enhances generalization by weighted averaging of model checkpoints. In summary, we present an efficient and effective framework for trainable weight averaging. 


## Dependencies

Install required dependencies:

```[bash]
pip install -r requirements.txt
```

## How to run

### Training from scratch
First, perform standard base training and save the checkpoints after each training epoch. Next, apply TWA-v/t to the solutions from the initial stage of training to enhance training efficiency. Below are some sample scripts:

CIFAR experiments:
```[bash]
bash run_cifar.sh
```
ImageNet experiments:
```[bash]
bash run_imagenet.sh
```


### Fine-tuning
First, fine-tune multiple models with different hyper-parameters and store them into the directory `model_dir`. As example, you can download the public avaliable checkpoints following the [code](https://github.com/mlfoundations/model-soups/blob/d5398f181ea51c5cd9d95ebacc6ea7132bb108ec/main.py#L67) from the [Model soup repo](https://github.com/mlfoundations/model-soups/tree/main) (72 CLIP ViT-B/32 models fine-tuned on ImageNet). 

Then run the TWA-v training under 4-bit quantization:

```[bash]
bash twa_soup.sh
```

## Citation
If you find this work helpful, please cite:
```
@misc{li2025trainableweightaveragingaccelerating,
      title={Trainable Weight Averaging: Accelerating Training and Improving Generalization}, 
      author={Tao Li and Zhehao Huang and Yingwen Wu and Zhengbao He and Qinghua Tao and Xiaolin Huang and Chih-Jen Lin},
      year={2025},
      eprint={2205.13104},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2205.13104}, 
}

@inproceedings{
    li2023trainable,
    title={Trainable Weight Averaging: Efficient Training by Optimizing Historical Solutions},
    author={Tao Li and Zhehao Huang and Qinghua Tao and Yingwen Wu and Xiaolin Huang},
    booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
    year={2023},
    url={https://openreview.net/forum?id=8wbnpOJY-f}
}
```