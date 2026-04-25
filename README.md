# Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition

[![license](https://img.shields.io/badge/LICENSE-Apache-green)](https://github.com/chenshunpeng/FoL/blob/main/LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FoL-orange)](https://huggingface.co/shunpeng/FoL)
[![arXiv](https://img.shields.io/badge/arXiv-2504.09881-red)](https://arxiv.org/abs/2504.09881)
[![star](https://img.shields.io/github/stars/chenshunpeng/FoL)](https://github.com/chenshunpeng/FoL)

This is the official repository for the AAAI 2025 paper "FoL" available at [AAAI Paper Page](https://ojs.aaai.org/index.php/AAAI/article/view/32811). In addition, our paper and its extensive supplementary materials can be found on [arXiv](https://arxiv.org/abs/2504.09881).

## Summary

We introduce Focus on Local **(FoL)**, a two-stage Visual Place Recognition (VPR) approach that enhances image retrieval and re-ranking by identifying and leveraging **reliable discriminative local regions**. Our method introduces three key contributions:

- **Reliable Discriminative Region Modeling**: We propose two novel loss functions—**Extraction-Aggregation Spatial Alignment Loss (SAL)** and **Foreground-Background Contrast Enhancement Loss (CEL)**—to explicitly learn discriminative local regions.
- **Weakly-Supervised Local Feature Learning**: We leverage pseudo-correspondences from aggregated global features to improve local matching supervision.
- **Efficient Re-ranking with Discriminative Region Guidance**: We use the learned discriminative regions to guide local feature matching, improving accuracy and efficiency.

You may refer to our **anonymous conference version** of the paper: [anonymous conference version](2104_Focus_on_Local_Finding_Re.pdf)

<img src="image/pipeline.jpg" width="800px">

## Setup & Requirements
**Quick install:**
```bash
# create and activate conda env
conda create -n fol python=3.9.19 -y
conda activate fol

# install dependencies
pip install -r requirements.txt
```
**Key dependencies:**
```
torch==2.0.0
torchvision==0.15.1
faiss-gpu==1.7.2
scikit-learn==1.3.0
numpy==1.26.4
opencv-python==4.10.0.84
```
> **Note — reproducibility:** The reranking step is sensitive to small numerical differences across `faiss-gpu`, `torch`, and `numpy` versions. Use the exact versions (in [requirements.txt](https://github.com/chenshunpeng/FoL/blob/main/requirements.txt)) to match paper results.

Install the Hugging Face Hub client (if you want to pull weights directly):

```bash
pip install huggingface_hub
```

## Download Pretrained Weights

You can download our pretrained FoL model either via Google Drive or directly from Hugging Face:

- **Google Drive (single files):**
  - FoL (ViT-L, FoL_large.pth): [link](https://drive.google.com/file/d/1-7LE_4Q0zL3S8lGVEH0Ob1NCFXq4KfJ8/view?usp=sharing)
  - FoL (ViT-B, FoL_base.pth): [link](https://drive.google.com/file/d/1Z05ZLFliQXOPJMH1YPdXqYjzC15-0nam/view?usp=sharing)

- **Google Drive (all models):** Shared folder "FoL_Trained_Models": [link](https://drive.google.com/drive/folders/1d3uEHdnzgWbGnj2g1ffLzLLI3pKuV7Vz?usp=sharing)

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

> **Note:** Hugging Face Hub downloads the ViT-L checkpoint (`FoL_large.pth`) by default. For ViT-B (`FoL_base.pth`), use the Google Drive links above.

---

## Evaluation

Assuming you have your datasets under `/datasets/` and your weights in `/weights/FoL.pth`:

```bash
python eval.py \
  --eval_datasets_folder=/datasets/ \
  --dataset_names pitts30k amstertime \
  --resume=/weights/FoL.pth
```

## Train

```bash
python train.py --eval_datasets_folder=.../datasets/ --eval_dataset_name pitts30k --epochs_num=8 --train_batch_size=60 --lr=6e-5 --optim=adamw --resize 322 322 --save_dir train_log/
```

## Performance
 
This table lists Recall@1, R@5, and R@10 for FoL across common VPR datasets, showing both **FoL-global** and **FoL-reranking** results. The best R@1 value within each dataset row is typeset in **bold**. All results are reported for ViT-L at 504×504 resolution; in the table, `G` and `R` denote `global` and `re-ranking`, respectively.

| ID | Dataset | G R@1 | G R@5 | G R@10 | │ | R R@1 | R R@5 | R R@10 | │ | ID | Dataset | G R@1 | G R@5 | G R@10 | │ | R R@1 | R R@5 | R R@10 |
|---:|:--|---:|---:|---:|:--:|---:|---:|---:|:--:|---:|:--|---:|---:|---:|:--:|---:|---:|---:|
| 1 | Pitts250k-test | 96.5 | 99.1 | 99.5 | │ | **97.0** | 99.2 | 99.5 | │ | 2 | MSLS-val | 93.1 | 96.9 | 97.4 | │ | **93.5** | 96.9 | 97.6 |
| 3 | MSLS-challenge | 78.7 | 90.8 | 93.0 | │ | **80.0** | 90.9 | 93.0 | │ | 4 | Tokyo24/7 | 96.2 | 98.7 | 98.7 | │ | **98.4** | 99.1 | 99.4 |
| 5 | Pitts30k | 93.9 | 97.2 | 98.1 | │ | **94.5** | 97.4 | 98.2 | │ | 6 | Sped | **92.1** | 96.5 | 98.0 | │ | 91.8 | 96.5 | 97.4 |
| 7 | Amstertime | 64.6 | 84.3 | 88.2 | │ | **70.1** | 89.0 | 91.8 | │ | 8 | Eynsham | 91.7 | 95.3 | 96.2 | │ | **92.4** | 95.8 | 96.6 |
| 9 | Nordland* | 78.3 | 90.8 | 94.0 | │ | **85.8** | 94.9 | 96.8 | │ | 10 | Nordland** | 87.8 | 94.5 | 96.4 | │ | **92.6** | 96.9 | 98.0 |
| 11 | SF-XL Night | 53.4 | 65.9 | 71.7 | │ | **60.5** | 72.8 | 75.8 | │ | 12 | SF-XL Occlusion | 51.3 | 65.8 | 73.7 | │ | **61.8** | 77.6 | 77.6 |
| 13 | SVOX | 98.4 | 99.4 | 99.6 | │ | **98.9** | 99.6 | 99.7 | │ | 14 | SVOX Sun | 98.1 | 99.4 | 99.5 | │ | **98.8** | 99.8 | 99.9 |
| 15 | SVOX Night | 98.3 | 99.6 | 99.6 | │ | **98.8** | 99.8 | 99.9 | │ | 16 | SVOX Snow | 99.1 | 99.7 | 99.8 | │ | **99.3** | 99.8 | 99.9 |
| 17 | SVOX Overcast | 97.9 | 99.2 | 99.3 | │ | **98.3** | 99.3 | 99.7 | │ | 18 | SVOX Rain | 96.5 | 99.6 | 99.7 | │ | **98.2** | 99.9 | 99.9 |

## Related Work
Our another ICLR 2026 work (single-stage VPR based on DINOv2) [SAGE](https://openreview.net/forum?id=DCpbEXqPvS) achieved SOTA performance on several datasets. The code is released at [here](https://github.com/chenshunpeng/SAGE).


## Acknowledgements
This code is based on the excellent work of:
 - [SelaVPR](https://github.com/Lu-Feng/SelaVPR), [CricaVPR](https://github.com/Lu-Feng/CricaVPR)
 - [SALAD](https://github.com/serizba/salad)
 - [Visual Geo-localization benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark), [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader)
 - [GSV-Cities](https://github.com/amaralibey/gsv-cities), [MixVPR](https://github.com/amaralibey/MixVPR)

## Citation

If you find this repo useful for your research, please cite the paper

```
@inproceedings{FoL,
  title={Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition},
  author={Wang, Changwei and Chen, Shunpeng and Song, Yukun and Xu, Rongtao and Zhang, Zherui and Zhang, Jiguang and Yang, Haoran and Zhang, Yu and Fu, Kexue and Du, Shide and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7536--7544},
  year={2025}
}
```
