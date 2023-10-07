<!-- PROJECT LOGO -->
<p align="center">
  <h3 align="center">Survey on Deep Face Restoration: From Non-blind to Blind and Beyond </h3>
  <p align="center">A comprehensive statistic on methods related to deep face restoration.
    <br />
    <a href="http://export.arxiv.org/pdf/2309.15490">[Paper]</a> &emsp;
    <a href="https://github.com/24wenjie-li/Awesome-Face-Restoration/blob/main/imgs/Supplementary.pdf">[Supplementary Material]</a>
  </p>
</p>

<p align="center">
  <img src="imgs/Non-blind.png">
</p>

|[<img src="imgs/Synthetic_DFDNet.png" height="131px"/>](https://imgsli.com/MjEwOTA4) | [<img src="imgs/Synthetic_GFPGAN.png" height="131px"/>](https://imgsli.com/MjEwOTA5) | [<img src="imgs/Synthetic_GCFSR.png" height="131px">](https://imgsli.com/MjEwOTEz) | [<img src="imgs/Synthetic_VGFR.png" height="131px"/>](https://imgsli.com/MjEwOTEy) | [<img src="imgs/Synthetic_CodeFormer.png" height="131px"/>](https://imgsli.com/MjEwOTEw)

|[<img src="imgs/Real_HiFaceGAN.png" height="131px"/>](https://imgsli.com/MjEwOTIx) | [<img src="imgs/Real_GFPGAN.png" height="131px"/>](https://imgsli.com/MjEwOTE4) | [<img src="imgs/Real_SGPN.png" height="131px">](https://imgsli.com/MjEwOTE3) | [<img src="imgs/Real_RestoreFormer.png" height="131px"/>](https://imgsli.com/MjEwOTE5) | [<img src="imgs/Real_DMDNet.png" height="131px"/>](https://imgsli.com/MjEwOTIw)

:boom: **Note**: More visual comparisons can be found in the <a href="http://export.arxiv.org/pdf/2309.15490">Paper</a> and <a href="https://github.com/24wenjie-li/Awesome-Face-Restoration/blob/main/imgs/Supplementary.pdf">Supplementary Material</a>.
  
---

## :clipboard: Citation

```
@article{li2023survey,
  title={Survey on Deep Face Restoration: From Non-blind to Blind and Beyond},
  author={Li, Wenjie and Wang, Mei and Zhang, Kai and Li, Juncheng and Li, Xiaoming and Zhang, Yuhang and Gao, Guangwei and Deng, Weihong and Lin, Chia-Wen},
  journal={arXiv preprint arXiv:2309.15490},
  year={2023}
}
```

## Table of contents
<!-- - [Survey paper](#survey-paper)
- [Table of contents](#table-of-contents) -->
- [Non-Blind Tasks](#non-blind-face-restoration)
- [Blind Tasks](#blind-face-restoration)
- [Joint Tasks](#joint-tasks)
  - [Joint Face Completion](#Joint-Face-Completion)
  - [Joint Face Frontalization](#Joint-Face-Frontalization)
  - [Joint Face Alignment](#Joint-Face-Alignment)
  - [Joint Face Recogntion](#Joint-Face-Recogntion)
  - [Joint Face Illumination Compensation](#Joint-Face-Illumination-Compensation)
  - [Joint Face Fairness](#Joint-Face-Fairness)
  - [Joint 3D Face Reconstruction](#Joint-3D-Face-Reconstruction)
- [Perfermance](#perfermance)
  - [Non-Bind Face Super-Resoution](#Non-blind-Face-Super-Resoution)
  - [Blind Face Restoration](#Blind-Face-Restoration)
  - [Blind Face Super-Resoution](#Blind-Face-Super-Resoution)
- [Benchmarks](#benchmarks)
  - [Datasets](#datasets)
  - [Losses](#loss)
 

## Non-Blind Tasks
|Pub<div style="width:60px">|Paper<div style="width:600px">|Technology<div style="width:100px">|
|:---:|:----:|:----:|
|AAAI2015|<a href="https://ojs.aaai.org/index.php/AAAI/article/download/9795/9654">Learning Face Hallucination in the Wild|CNN-based|
|ECCV2016|<a href="https://browse.arxiv.org/pdf/1607.05046v1.pdf">Deep cascaded bi-network for face hallucination|CNN-based|
|ECCV2016|<a href="https://browse.arxiv.org/pdf/1707.00737.pdf">Ultra-resolving face images by discriminative generative networks|GAN-based|
|ICCV2017|<a href="https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Wavelet-SRNet_A_Wavelet-Based_ICCV_2017_paper.pdf">Wavelet-SRNet: A Wavelet-based CNN for Multi-scale Face Super Resolution|Wavelet transform|
|CVPR2017|<a href="https://arxiv.org/abs/1708.03132">Attention-Aware Face Hallucination via Deep Reinforcement Learning|Attention-based|
|CVPR2017|<a href="https://ieeexplore.ieee.org/document/8100053">Hallucinating very low-resolution unaligned and noisy face images by transformative discriminative autoencoders|GAN-based|
|IJCAI2017|<a href="https://arxiv.org/pdf/1708.00223.pdf">Learning to Hallucinate Face Images via Component Generation and Enhancement|Prior-based|
|ICASSP2017|<a href="https://ieeexplore.ieee.org/document/8462170">Face Hallucination Based on Key Parts Enhancement|CNN-based|
|CVPR2018|<a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_FSRNet_End-to-End_Learning_CVPR_2018_paper.pdf">FSRNet: End-to-End Learning Face Super-Resolution with Facial Priors|Prior-based|
|CVPRW2018|<a href="https://arxiv.org/pdf/1811.02328.pdf">Attribute Augmented Convolutional Neural Network for Face Hallucination|GAN-based|
|ECCV2018|<a href="https://arxiv.org/pdf/1811.02328.pdf">Super-Identity Convolutional Neural Network for Face Hallucination|Prior-based|
|ECCV2018|<a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Xin_Yu_Face_Super-resolution_Guided_ECCV_2018_paper.pdf">Face Super-resolution Guided by Facial Component Heatmaps|Prior-based|
|IJCAI2018|<a href="https://arxiv.org/pdf/1806.10726.pdf">Deep CNN Denoiser and Multi-layer Neighbor Component Embedding for Face Hallucination|CNN-based|
|ICASSP2018|<a href="https://ieeexplore.ieee.org/document/8462170">FACE HALLUCINATION BASED ON KEY PARTS ENHANCEMENT|Prior-based|
|CVPR2019|<a href="https://arxiv.org/pdf/1806.10726.pdf">Exemplar Guided Face Image Super-Resolution without Facial Landmarks|Prior/GAN-based|
|BMVC2019|<a href="https://arxiv.org/pdf/1908.08239.pdf">Progressive Face Super-Resolution via Attention to Facial Landmark|Prior/GAN-based|
|AAAI2019|<a href="https://ojs.aaai.org/index.php/AAAI/article/view/4937#:~:text=In%20this%20paper%2C%20we%20present%20a%20novel%20deep,a%20multi-block%20cascaded%20structure%20network%20with%20dense%20connection.">Residual Attribute Attention Network for Face Image Super-Resolution|Attention-based|
|TMM2019|<a href="https://ieeexplore.ieee.org/document/8936424">ATMFN: Adaptive-threshold-based Multi-model Fusion Network for Compressed Face Hallucination|CNN/GAN/RNN-based|
|IJCV2019|<a href="https://arxiv.org/pdf/1811.09019.pdf">Joint Face Hallucination and Deblurring via Structure Generation and Detail Enhancement|Prior-based|
|PR2019|<a href="https://www.sciencedirect.com/science/article/abs/pii/S003132031930202X">Face hallucination from low quality images using definition-scalable inference|Frequency-based|
|TIP2020|<a href="https://browse.arxiv.org/pdf/2012.01211.pdf">Learning Spatial Attention for Face Super-Resolution|Attention-based|
|TIP2020|<a href="https://ieeexplore.ieee.org/document/9082831/metrics#metrics">Deblurring Face Images using Uncertainty Guided
Multi-Stream Semantic Networks|Prior-based|
|TMM2020|<a href="https://ieeexplore.ieee.org/document/9055090">Learning Face Image Super-Resolution through Facial Semantic Attribute Transformation and Self-Attentive Structure Enhancement|Prior-based|
|ECCV2020|<a href="https://browse.arxiv.org/pdf/2007.09454.pdf">Face Super-Resolution Guided by 3D Facial Priors|Prior-based|
|CVPR2020|<a href="https://arxiv.org/abs/2003.13063">Deep Face Super-Resolution with Iterative Collaboration between Attentive Recovery and Landmark Estimation|Prior-based|
|AAAI2020|<a href="https://browse.arxiv.org/pdf/2002.06518.pdf">Facial Attribute Capsules for Noise Face Super Resolution|Prior-based|
|ICASSP2020|<a href="https://ieeexplore.ieee.org/document/9053398">PARSING MAP GUIDED MULTI-SCALE ATTENTION NETWORK FOR FACE HALLUCINATION|Prior-based|
|WACV2020|<a href="https://ieeexplore.ieee.org/document/9093399">Component Attention Guided Face Super-Resolution Network: CAGFace|Prior-based|





## Blind Tasks
|Pub<div style="width:60px">|Paper<div style="width:600px">|Technology<div style="width:100px">|
|:---:|:----:|:----:|
|ECCV2018|<a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaoming_Li_Learning_Warped_Guidance_ECCV_2018_paper.pdf">Learning Warped Guidance for Blind Face Restoration|CNN-based|
|ECCV2018|<a href="https://arxiv.org/abs/1705.09966">Attribute-guided face generation using conditional cyclegan|CycleGAN-based|
|CVPR2018|<a href="https://browse.arxiv.org/pdf/1712.02765.pdf">Super-fan: Integrated facial landmark localization and super-resolution of real-world low resolution faces in arbitrary poses with gans|Prior/GAN-based|
|CVPR2018|<a href="https://arxiv.org/abs/1803.03345">Exploiting Semantics for Face Image Deblurring|Prior-based|
|CVPRW2018|<a href="https://ieeexplore.ieee.org/document/8575269">Learning Face Deblurring Fast and Wide|Prior-based|
|ICIP2019|<a href="https://ieeexplore.ieee.org/document/8803393">GUIDED CYCLEGAN VIA SEMI-DUAL OPTIMAL TRANSPORT FOR PHOTO-REALISTIC FACE SUPER-RESOLUTION|CycleGAN-based|
|IJCV2020|<a href="https://arxiv.org/abs/2001.06822v2">Exploiting Semantics for Face Image Deblurring|Prior-based|


## :e-mail: Contact

If you have any question, please email `lewj2408@gmail.com`
