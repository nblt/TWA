# TWA
The code is the official implementation of our ICLR paper 
[Trainable Weight Averaging: Efficient Training by Optimizing Historical Solutions](https://openreview.net/pdf?id=8wbnpOJY-f). 

We propose to conduct neural network training in a tiny subspace spanned by historical solutions. Such optimization is equivalent to performing weight averaging on these solutions with trainable coefficients (TWA), in contrast with the equal averaging coefficients as in [SWA](https://github.com/timgaripov/swa). We show that TWA is able to achieve great training efficiency by optimizing historical solutions and also provide an efficient and scalable framework for multi-node training. Besides, TWA is also able to improve finetune results from multiple training configurations, which we are currently focusing on. This [colab](https://colab.research.google.com/drive/1fxUJ0K8dd7V3gsozmKsHhfdYHhYVB-WZ?usp=sharing) provides an exploratory example we adapt from [Model Soups](https://github.com/mlfoundations/model-soups).


## Dependencies

Install required dependencies:

```
pip install -r requirements.txt
```

## How to run

### TWA in tail stage training
We first show that TWA could improve the performance of SWA in the original SWA setting, where the improvements are more significant when the tail learning rate is larger.
```
cd swa
```
First, run SWA using original [code](https://github.com/timgaripov/swa):
```
bash run.sh
```
Then, we could perform TWA using:
```
bash run_twa.sh
```
The training configuration is easy to set as you need in the scripts.

### TWA in head stage training
In this part, we conduct TWA in the head training stage, where we achieve considerably **30%-40%** epochs saving on CIFAR-10/100 and ImageNet, with a comparable or even better performance against regular training.
We show sample usages in `run.sh`:

```
bash run.sh
```


## Citation
If you find this work helpful, please cite:
```
@inproceedings{
    li2023trainable,
    title={Trainable Weight Averaging: Efficient Training by Optimizing Historical Solutions},
    author={Tao Li and Zhehao Huang and Qinghua Tao and Yingwen Wu and Xiaolin Huang},
    booktitle={International Conference on Learning Representations},
    year={2023}
}
```