# Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition

[![license](https://img.shields.io/badge/LICENSE-Apache-green)](https://github.com/chenshunpeng/FoL/blob/main/LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FoL-orange)](https://huggingface.co/shunpeng/FoL)
[![arXiv](https://img.shields.io/badge/arXiv-2504.09881-red)](https://arxiv.org/abs/2504.09881)
[![star](https://img.shields.io/github/stars/chenshunpeng/FoL)](https://github.com/chenshunpeng/FoL)

This is the official repository for the AAAI 2025 paper "FoL" available at [AAAI Paper Page](https://ojs.aaai.org/index.php/AAAI/article/view/32811). In addition, our paper and its extensive supplementary materials can be found on [arXiv](https://arxiv.org/abs/2504.09881).


## Summary

We introduce **Focus on Local (FoL)**, a two-stage Visual Place Recognition (VPR) approach that enhances image retrieval and re-ranking by identifying and leveraging **reliable discriminative local regions**. Our method introduces three key contributions:

- **Reliable Discriminative Region Modeling**: We propose two novel loss functions—**Extraction-Aggregation Spatial Alignment Loss (SAL)** and **Foreground-Background Contrast Enhancement Loss (CEL)**—to explicitly learn discriminative local regions.
- **Weakly-Supervised Local Feature Learning**: We leverage pseudo-correspondences from aggregated global features to improve local matching supervision.
- **Efficient Re-ranking with Discriminative Region Guidance**: We use the learned discriminative regions to guide local feature matching, improving accuracy and efficiency.

You may refer to our **anonymous conference version** of the paper: [anonymous conference version](2104_Focus_on_Local_Finding_Re.pdf)

<img src="image/pipeline.jpg" width="800px">

## Setup

Tested on **Pytorch 2.0.0** with **CUDA 11.7**. To set up the environment:

```bash
conda env create -f environment.yml
```

Install the Hugging Face Hub client (if you want to pull weights directly):

```bash
pip install huggingface_hub
```

## Download Pretrained Weights

You can download our pretrained FoL model either via Google Drive or directly from Hugging Face:

- **Google Drive**  
  [here](https://drive.google.com/file/d/1-7LE_4Q0zL3S8lGVEH0Ob1NCFXq4KfJ8/view?usp=sharing)

- **Hugging Face Hub**  
  ```python
  from huggingface_hub import hf_hub_download

  # this will download FoL.pth into your cache folder
  FoLpath = hf_hub_download(
      repo_id="shunpeng/FoL",
      filename="FoL.pth"
  )
  print("Downloaded weights to:", FoLpath)
  ```

---

## Evaluation

Assuming you have your datasets under `/datasets/` and your weights in `/weights/FoL.pth`:

```bash
python eval.py \
  --eval_datasets_folder=/datasets/ \
  --dataset_names pitts30k amstertime \
  --resume=/weights/FoL.pth
```

## To-do
- [x] Public release of evaluation code and pretrained FoL model  
- [ ] Public release of the training code (coming soon)  
- [ ] More detailed documentation (coming soon)

## Acknowledgements
This code is based on the excellent work of:
 - [DINOv2](https://github.com/facebookresearch/dinov2)
 - [AnyLoc](https://github.com/AnyLoc/AnyLoc)
 - [SelaVPR](https://github.com/Lu-Feng/SelaVPR)
 - [CricaVPR](https://github.com/Lu-Feng/CricaVPR)
 - [SALAD](https://github.com/serizba/salad)

## Citation

If you find this repo useful for your research, please cite the paper

```
@inproceedings{wang2025focus,
  title={Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition},
  author={Wang, Changwei and Chen, Shunpeng and Song, Yukun and Xu, Rongtao and Zhang, Zherui and Zhang, Jiguang and Yang, Haoran and Zhang, Yu and Fu, Kexue and Du, Shide and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7536--7544},
  year={2025}
}
```