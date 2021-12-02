# [AAAI 2022] Low-Light Image Enhancement with Normalizing Flow
### [Paper](!https://arxiv.org/pdf/2109.05923.pdf) | [Project Page](!https://wyf0912.github.io/LLFlow/)

**Low-Light Image Enhancement with Normalizing Flow**
<br>_Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-pui Chau, Alex C. Kot_<br>
In AAAI'2022

## Overall
![Framework](images/framework.png)

### Quantitative results
The evauluation results on LOL are as follows
| Method | PSNR | SSIM | LPIPS |
| :-- | :--: | :--: | :--: |
| LIME | 16.76 | 0.56 | 0.35 |
| RetinexNet | 16.77 | 0.56 | 0.47 |
| DRBN | 20.13 | 0.83 | 0.16 | 
| Kind | 20.87 | 0.80 | 0.17 |
| KinD++ | 21.30 | 0.82 | 0.16 |
| **LLFlow (Ours)** | **25.19** | **0.93** | **0.11** |

and th
### Visual Results
![Visual comparison with state-of-the-art low-light image enhancement methods on LOL dataset.](images/Input_778-Reference_778.png)

## Get Started
### Dependencies and Installation
- Python 3.8
- Pytorch 1.9

1. Clone Repo
```
git clone https://github.com/wyf0912/LLFlow.git
```
2. Create Conda Environment
```
conda create --name LLFlow python=3.7
conda activate LLFlow
```
3. Install Dependencies
```
cd 
```

### Pretrained Model

### Test

### Train

## Citation
If you find our work useful for your research, please cite our paper
```
@article{wang2021low,
  title={Low-Light Image Enhancement with Normalizing Flow},
  author={Wang, Yufei and Wan, Renjie and Yang, Wenhan and Li, Haoliang and Chau, Lap-Pui and Kot, Alex C},
  journal={arXiv preprint arXiv:2109.05923},
  year={2021}
}
```
## Contact
If you have any question, please feel free to contact us via yufei001@ntu.edu.sg.