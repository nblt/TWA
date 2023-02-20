### TWA in tail stage training
We show that TWA could improve the performance of SWA in the original SWA setting, where the improvements are more significant when the tail learning rate is larger.
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