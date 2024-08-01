<div align="center">
  <img src="images/image_no_back.png" width="200" height="200">
  <h1>  🔥 LLaVA-MORE 🔥
    
 Enhancing Visual Instruction Tuning with LLaMA 3.1
  </h1>  

[![HuggingFace](https://img.shields.io/badge/🤗_LLaVA_MORE-1d8c0a)](https://huggingface.co/collections/aimagelab/llava-more-66aa6c49167e190bf27e7be4)
[![HuggingFace](https://img.shields.io/badge/🤗_AImageLab_-white)](https://huggingface.co/aimagelab)
[![Website](https://img.shields.io/badge/AImageLab-red)](https://aimagelab.ing.unimore.it/imagelab)

</div>


<div align='center'>

#### [Federico Cocchi](https://federico1-creator.github.io/Federico_Cocchi/), [Nicholas Moratelli](https://github.com/NicholasMoratelli), [Davide Caffagni](https://github.com/dcaffo98), [Sara Sarto](https://github.com/sarasarto),
#### [Marcella Cornia](https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=90), [Lorenzo Baraldi](https://www.lorenzobaraldi.com/), and [Rita Cucchiara](https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=1)

</div>

## Citation
If you make use of our work, please cite our repo:

```bibtex
@misc{cocchi2024llavamore,
      title={{LLaVA-MORE: Enhancing Visual Instruction Tuning with LLaMA 3.1}},
      author={Federico, Cocchi and Nicholas, Moratelli and Davide, Caffagni and Sara, Sarto and Marcella, Cornia and Lorenzo, Baraldi and Rita, Cucchiara},
      url={https://github.com/aimagelab/LLaVA-MORE},
      year={2024}
}
```



## 📢 Latest Updates
- [2024/08/01] 🔥 First release of our LLaVA-MORE 8B model, based on LLaMA 3.1.
- [2024/08/01] 🔎 If you are interested in this area of research, check out [our survey](https://arxiv.org/abs/2402.12451) on the revolution of multimodal LLMs, recently published in ACL (Findings).
- [2024/08/01] 📚 Check out the latest researches from [AImageLab](https://aimagelab.ing.unimore.it/imagelab/).

## Table of Contents

1. [Overview](#overview)
2. [Performance](#performance)
3. [Checkpoints](#checkpoints)
4. [Installation](#installation)
5. [Training](#training)
6. [Inference](#inference)
7. [Acknowledgments](#acknowledgments)

## Overview

```LLaVA-MORE``` enhances the well-known LLaVA architecture by integrating for the first time the use of LLaMA 3.1 as the language model. We are publicly releasing the checkpoints for stages one and two for the first model with 8B parameters.

To further support the research community in enhancing Multimodal LLM performance, we are also releasing the training code and scripts for distributed training.

Remember to star the repository to stay updated on future releases 🤗 and try our models [here](http://www.more.unimore.it/)!

## Performance
In this section, we present the performance of our model compared to other versions of LLaVA across different multimodal datasets.

<div align="center">
<img src="images/radar_plot.png" width="500"">
</div>

### Benchmarks and Comparisons on Instrucion Multimodal Datasets in the Literature

<div align="center">

|       Model Name     |  Text-VQA*  |  Science-QA  |  AI2D  |  SEED-vid  |  SEED-all  |  SEED-img  |  MMMU  |  MMBench-Cn  |  MMBench-En  |  POPE  |  GQA  |   MME-P  |  MME-C  |
|----------------------|:----------: |:------------:|:------:|:----------:|:----------:|:----------:|:------:|:------------:|:------------:|:------:|:-----:|:--------:|:-------:|
|    LLaVA-v1.5-7B     |    58.2     |     69.0     |  56.4  |    42.0    |    61.6    |    66.8    |  34.2  |      56.5    |      65.3    |  **85.6**  |  62.4 |  1474.3  |  314.6  |
| LLaVA-v1.5-LLaMA3-8B |    57.6     |     74.2     |  60.7  |    42.0    |    **64.3**    |    **70.1**    |  37.3  |      65.4    |      70.3    |  85.4  |  63.5 |  **1544.4**  |  330.3  |
|  **LLaVA-MORE-8B**   |    **58.4**    |     **76.3**     |  **61.8**  |    **42.4**    |    64.1    |    69.8    |  **39.4**  |      **68.2**    |      **72.4**    |  85.1  |  **63.6** |  1531.5  |  **353.3**  |
</div>

*\* the results of TextVQA are calculated with OCR token in the input prompt.*

## Checkpoints

In the table below, you can find links to ours 🤗 Hugging Face models.

|         Model Name        |      🤗 Hugging Face      |             Summary                            |
|---------------------------|:-------------------------:|------------------------------------------------|
| LLaVA_MORE-llama_3_1-8B-pretrain | [Hugging Face Model](https://huggingface.co/aimagelab/LLaVA_MORE-llama_3_1-8B-pretrain)  | Pretrained on [LCS-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) and using [LLaMA 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) as LLM backbone            |
| LLaVA_MORE-llama_3_1-8B-finetuning | [Hugging Face Model](https://huggingface.co/aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning)  | Finetuned on [LLaVA-Instruct-665K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) and using [LLaMA 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) as LLM backbone         |


## Installation
To create the conda environment named ```more``` use the following instructions.
With this environment you will have all the packages to run the code in this repo. 
```
conda create -n more python==3.8.16
conda activate more
pip install -r requirements.txt
```

Note that the requirements are heavily inspired by the original [LLaVA](https://github.com/haotian-liu/LLaVA.git) repo.

## Training
To help the community in training complex systems in distributed scenarios, we are publicly releasing not only the source code but also the bash scripts needed to train ```LLaVA-MORE``` on HPC facilities with a SLURM scheduler.

To further extend the reproducibility of our approach, we are also releasing the [wandb logs](https://api.wandb.ai/links/aimagelab/kq668y5l) of the training runs.

**Pretraining**

``` bash
sbatch scripts/more/11_pretrain_llama_31_acc_st_1.sh
```
**Finetuning**
``` bash
sbatch scripts/more/12_finetuning_llama_31_acc_st_1.sh
```

### Visual Backbones

As mentioned before, ```LLaVA-MORE``` introduces the use of LLaMA 3.1 within the LLaVA architecture for the first time. However, this repository goes beyond that single enhancement.
We have also incorporated the ability to use different visual backbones, such as SigLIP, and various methods for managing image resolutions (S2). Additionally, we have experimented with different data mixtures to stress data quality during the LLaVA training stages.

Considering that, you can view this repos as an effort to expand the study of Multimodal LLMs in multiple directions and as a 
starting point for enhancing new features to improve the connection between images and language.

You can find more references in this folder: ```scripts/more```


## Inference
You can try our ```LLaVA-MORE``` in the Image-To-Text task by running the following script.
``` python
python -u llava/eval/run_llava.py
```
If you get out-of-memory problems, consider loading the model weights in 8 bit (```load_in_8bit=True```).

## Acknowledgments
We thank the [LLaVA](https://github.com/haotian-liu/LLaVA.git) team for open-sourcing a modular codebase to extend and train different models within the LLaVA family.
We are also happy users of the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval.git) library, which has significantly reduced the evaluation time of our checkpoints across different datasets.

We also thank [CINECA](https://www.hpc.cineca.it/systems/hardware/leonardo/) for the availability of high-performance computing resources used to train ```LLaVA-MORE```. This work is supported by the PNRR-M4C2 project [FAIR - Future Artificial Intelligence Research](https://fondazione-fair.it/) and by the PNRR project [ITSERR - Italian Strengthening of Esfri RI Resilience](https://www.itserr.it/).


In case you face any issues or have any questions, please feel free to create an issue.
Additionally, we welcome you to open a pull request to integrate new features and contribute to our project.
