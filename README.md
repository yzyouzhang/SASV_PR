# SASV_PR

## A Probabilistic Fusion Framework for Spoofing Aware Speaker Verification
[Paper link](https://arxiv.org/pdf/2202.05253.pdf)

### Data preparation
Get the pre-trained ASV and CM embeddings from [here](https://drive.google.com/drive/folders/1kn_slob4BGHPmk_O8HaAiJE5P6qqBFV7?usp=sharing).
Our embedding extractions process is the same as in the [baseline repository](https://github.com/sasv-challenge/SASVC2022_Baseline) (commit 8bfbf1f3f7373). Please refer to their `save_embeddings.py`.

### Run our code
Our code is based on PyTorch.

Before running, you need to specify where you would like to save the model by `-o`, you also need to specify which model you want to use by `-m`, the default is `pr_s_f`.

The options for `-m` are: `pr_l_i`, `pr_s_i`, `pr_c_i`, `pr_l_f`, `pr_s_f` if you would like to use our methods. Please refer to our paper for more details.

```
python3 main_train.py -o ./exp_result/ -m pr_s_f
```

### Citation
```
@article{zhang2022prob,
  title={A Probabilistic Fusion Framework for Spoofing Aware Speaker Verification},
  author={Zhang, You and Zhu, Ge and Duan, Zhiyao},
  journal={arXiv preprint arXiv:2202.05253},
  year={2022}
}
```