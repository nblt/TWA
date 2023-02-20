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

We show sample usages in `run.sh`:

```
bash run.sh
```

## Citation
```
@inproceedings{
    li2023trainable,
    title={Trainable Weight Averaging: Efficient Training by Optimizing Historical Solutions},
    author={Tao Li and Zhehao Huang and Qinghua Tao and Yingwen Wu and Xiaolin Huang},
    booktitle={International Conference on Learning Representations},
    year={2023}
}
```