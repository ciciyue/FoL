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

- **Google Drive:** [here](https://drive.google.com/file/d/1-7LE_4Q0zL3S8lGVEH0Ob1NCFXq4KfJ8/view?usp=sharing)

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

## Train

```bash
python train.py --eval_datasets_folder=.../datasets/ --eval_dataset_name pitts30k --epochs_num=8 --train_batch_size=60 --lr=6e-5 --optim=adamw --resize 322 322 --save_dir train_log/
```

## Performance

This table lists Recall@1, R@5 and R@10 for our FoL model across common VPR datasets, showing results for both **FoL-global** and **FoL-reranking**. 
To facilitate immediate visual comparison, the best R@1 value within each dataset row is typeset in **bold**.

<table style="border-collapse: collapse; width: 70%; margin: 1.5em 0; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #f0f0f0;">
      <th rowspan="2" style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">ID</th>
      <th rowspan="2" style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">Dataset</th>
      <th colspan="3" style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">FoL-global</th>
      <th colspan="3" style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">FoL-reranking</th>
    </tr>
    <tr style="background-color: #f0f0f0;">
      <th style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">R@1</th>
      <th style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">R@5</th>
      <th style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">R@10</th>
      <th style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">R@1</th>
      <th style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">R@5</th>
      <th style="text-align: center; border: 1px solid #ddd; padding: 8px; font-weight: bold;">R@10</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">1</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">Pitts250k-test</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.5</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.1</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.5</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>97.0</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.2</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.5</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">2</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">MSLS-val</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">93.1</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">97.4</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>93.5</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">97.6</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">3</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">MSLS-challenge</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">78.7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">90.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">93.0</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>80.0</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">90.9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">93.0</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">4</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">Tokyo24/7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.2</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">98.7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">98.7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>98.4</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.1</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.4</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">5</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">Pitts30k</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">93.9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">97.2</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">98.1</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>94.5</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">97.4</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">98.2</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">6</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">Sped</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>92.1</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.5</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">98.0</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">91.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.5</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">97.4</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">Amstertime</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">64.6</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">84.3</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">88.2</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>70.1</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">89.0</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">91.8</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">Eynsham</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">91.7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">95.3</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.2</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>92.4</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">95.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.6</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">Nordland*</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">78.3</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">90.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">94.0</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>85.8</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">94.9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.8</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">10</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">Nordland**</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">87.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">94.5</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.4</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>92.6</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">98.0</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">11</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">SF-XL Night</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">53.4</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">65.9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">71.7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>60.5</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">72.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">75.8</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">12</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">SF-XL Occlusion</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">51.3</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">65.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">73.7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>61.8</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">77.6</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">77.6</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">13</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">SVOX</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">98.4</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.4</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.6</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>98.9</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.6</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.7</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">14</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">SVOX Sun</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">98.1</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.4</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.5</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>98.8</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.9</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">15</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">SVOX Night</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">98.3</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.6</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.6</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>98.8</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.9</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">16</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">SVOX Snow</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.1</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>99.3</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.8</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.9</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">17</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">SVOX Overcast</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">97.9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.2</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.3</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>98.3</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.3</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.7</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">18</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">SVOX Rain</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">96.5</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.6</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.7</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>98.2</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.9</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">99.9</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">19</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">St. Lucia</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>99.9</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">100.0</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">100.0</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;"><strong>99.9</strong></td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">100.0</td>
      <td style="text-align: center; border: 1px solid #ddd; padding: 8px;">100.0</td>
    </tr>
  </tbody>
</table>

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
