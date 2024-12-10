<p align="center">
  <img src="asset/logo.gif"  height=120>
</p> 

### <div align="center"> Learning Visual Generative Priors without Text<div> 
<div align="center">
<div style="text-align: center">
  <a href="https://scholar.google.com/citations?user=dNhzCu4AAAAJ&hl=zh-CN">Shuailei Ma*</a><sup>1</sup>,
  <a href="https://zkcys001.github.io/">Kecheng Zheng*</a><sup>2</sup>,
  <a href="https://ieeexplore.ieee.org/author/37836204100">Ying Wei✉️</a><sup>1</sup>,            <a href="https://weiwu-ww.github.io/">Wei Wu</a><sup>2</sup>, <a href="https://scholar.google.com/citations?user=ILpxpfwAAAAJ&hl=zh-CN">Fan Lu</a><sup>2</sup>,
  <a href="https://scholar.google.com/citations?hl=en&user=rQKkIykAAAAJ">Yifei Zhang</a><sup>3</sup>,<a href="https://scholar.google.com/citations?user=UHCDCRMAAAAJ&hl=en">Chen-Wei Xie</a><sup>4</sup>,
  <a href="https://scholar.google.com/citations?user=BwdpTiQAAAAJ&hl=zh-CN">Biao Gong</a><sup>2</sup>,
  <a href="https://scholar.google.com/citations?user=-ACBm-gAAAAJ&hl=zh-TW">Jiapeng Zhu</a><sup>5</sup>,
  <a href="https://shenyujun.github.io/">Yujun Shen✉️</a><sup>2</sup> <br>
  <sup>1</sup>Northeastern University, China <sup>2</sup>Ant Group <sup>3</sup>SJTU <sup>4</sup>Alibaba Group <sup>5</sup>HKUST <br>
  <sup>*</sup>equal contribution <sup>✉️</sup>corresponding author
</div> 
<br>
<div style="text-align: center;">
  <a href=""><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:Lumos&color=red&logo=arxiv"></a> &ensp;
  <a href="https://xiaomabufei.github.io/lumos/"><img src="https://img.shields.io/badge/Project-Website-blue"></a> &ensp;
  <a href="https://huggingface.co/Xiaomabufei/lumos"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=yellow"></a> &ensp;
</div>
</div> 

## 📝 Content
* [Update Log](#📣-update-log)
* [Abstract](#🪄✨-abstract)
* [Setup](#️⚙️-setup)
* [License](#🕊️-license)
* [Citation](#📖-citation)
* [Acknowledgement](#❤️-acknowledgement)


## 📣 Update Log
- [2024.11.21] 🎉 Here comes Lumos, we release the code and gradio demos of Lumos-I2I and Lumos-T2I. 

## 🪄✨ Abstract
<!-- <b>TL; DR: <font color="purple">Lumos</font> is a Transformer-based diffusion model.</b> -->

<details><summary>CLICK for the full abstract</summary>
Although text-to-image (T2I) models have recently thrived as visual generative priors, their reliance on high-quality text-image pairs makes scaling up expensive.
We argue that grasping the cross-modality alignment is not a necessity for a sound visual generative prior, whose focus should be on texture modeling.
Such a philosophy inspires us to study image-to-image (I2I) generation, where models can learn from in-the-wild images in a self-supervised manner.
We first develop a pure vision-based training framework, Lumos, and confirm the feasibility and the scalability of learning I2I models.
We then find that, as an upstream task of T2I, our I2I model serves as a more foundational visual prior and achieves on-par or better performance than existing T2I models using only 1/10 text-image pairs for fine-tuning.
We further demonstrate the superiority of I2I priors over T2I priors on some text-irrelevant visual generative tasks, like image-to-3D and image-to-video.
</details>

![Visualization various downstream tasks  of Lumos](asset/teaser.png)


## ⚙️ Setup
Follow the following guide to set up the environment.
- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.2.1+cu11.8](https://pytorch.org/)
- Better create a virtual environment

Install the required dependencies by following the command.

1. git clone repo.
    ```
    git clone https://github.com/xiaomabufei/lumos.git
    cd lumos
    ```
2. download model checkpoints
    ```
    mkdir ./checkpoints && cd ./checkpoints
    git lfs install
    git clone https://huggingface.co/Xiaomabufei/lumos
    ```

3. create environment
    ```
    conda create -n lumos python=3.9 -y
    conda activate lumos
    ```

4. install torch with GPU support
    ```
    pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
    ```

5. install xformers corresponding to torch and cuda
    ```
    pip install -U xformers==0.0.25
    ```

6. install the remaining environment
    ```
    pip install -r requirements.txt
    ```

7. run lumos Image Interpolation
    ```
    python gradio_demos/lumos_I2I.py
    ```

8. run lumos Text-to-Image Generation
    ```
    python gradio_demos/lumos_T2I.py
    ```
    If you are mainland user, you may try `export HF_ENDPOINT=https://hf-mirror.com` to use huggingface mirror to facilitate the download of some necessary checkpoints to run our system.

## 🕊️ License
This repository is released under the MiT license as found in the [LICENSE](LICENSE) file.

## 📖 Citation
Don't forget to cite this source if it proves useful in your research!
```bibtex
@article{Lumos2024, 
	title={Learning Visual Generative Priors without Text}, 
	author={Ma, Shuailei and Zheng, Kecheng and Wei, Ying and Wu, Wei and Lu, Fan and Zhang, Yifei and Xie, Chen-Wei and Gong, Biao and Zhu, Jiapeng and Shen, Yujun}, 
	year={2024}, 
	eprint={arxiv}, 
	archivePrefix={arXiv}, 
	primaryClass={cs.CV}}
```


# ❤️ Acknowledgement
<!-- ## 🤗 <a name="acknowledgement"></a>Acknowledgement -->
Our implementation is based on [DiT](https://github.com/nullquant/ComfyUI-BrushNet), [Pixart-α](https://github.com/facebookresearch/DiT) and [Dino](https://github.com/facebookresearch/dino). Thanks for their remarkable contribution and released code!
