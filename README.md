# Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition
This is the official repository for the AAAI 2025 paper "Focus on Local: Finding Reliable Discriminative Regions for Visual Place Recognition".

## Summary

We introduce **Focus on Local (FoL)**, a two-stage Visual Place Recognition (VPR) approach that enhances image retrieval and re-ranking by identifying and leveraging **reliable discriminative local regions**. Our method introduces three key contributions:

- **Reliable Discriminative Region Modeling**: We propose two novel loss functions—**Extraction-Aggregation Spatial Alignment Loss (SAL)** and **Foreground-Background Contrast Enhancement Loss (CEL)**—to explicitly learn discriminative local regions.
- **Weakly-Supervised Local Feature Learning**: We leverage pseudo-correspondences from aggregated global features to improve local matching supervision.
- **Efficient Re-ranking with Discriminative Region Guidance**: We use the learned discriminative regions to guide local feature matching, improving accuracy and efficiency.

You can also read our full paper for more details: [Download the Paper (2104_Focus_on_Local_Finding_Re.pdf)](2104_Focus_on_Local_Finding_Re.pdf)

<img src="image/pipeline.jpg" width="800px">

## Setup

Tested on **Pytorch 2.0.0** with **CUDA 11.7**. To set up the environment:

```bash
conda env create -f environment.yml
```

## Evaluation

Download a pretrained **FoL model** from [here](https://drive.google.com/file/d/1-7LE_4Q0zL3S8lGVEH0Ob1NCFXq4KfJ8/view?usp=sharing). Evaluate with:

```bash
python eval.py --eval_datasets_folder=/datasets/ --dataset_names pitts30k amstertime --resume=/weights/FoL.pth
```

## To-do
- [x] Public release of evaluation code and pretrained FoL model  
- [ ] Public release of the training code (coming soon)  
- [ ] More detailed documentation (coming soon)

## Acknowledgements
This code is based on the excellent work of:
 - [AnyLoc](https://github.com/AnyLoc/AnyLoc)
 - [SelaVPR](https://github.com/Lu-Feng/SelaVPR)
 - [CricaVPR](https://github.com/Lu-Feng/CricaVPR)
 - [SALAD](https://github.com/serizba/salad)
