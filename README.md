<div align="center">
  <!-- <h1><b> ReProTS </b></h1> -->
  <!-- <h2><b> ReProTS </b></h2> -->
  <h2><b> ReProTS: Reprogrammed Prompting for Time Series Forecasting with Frozen Large Language Models </b></h2>


<div align="center">



</div>

<p align="center">

<img src="./figures/model1.png" width="90%">

</p>

---
>
>  Please let us know if you find out a mistake or have any suggestions!
> 
>  If you find this resource helpful, please consider to star this repository and cite our research:

```

## Requirements
Use python 3.11 from MiniConda

- torch==2.2.2
- accelerate==0.28.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.14.0
- sentencepiece==0.2.0

To install all dependencies:
```
pip install -r requirements.txt


```

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing), then place the downloaded contents under `./dataset`

## Quick Demos
1. Download datasets and place them under `./dataset`
2. Tune the model. We provide five experiment scripts for demonstration purpose under the folder `./scripts`. For example, you can evaluate on ETT datasets by:
```

```bash
bash ./scripts/ReProTS_ETTh1.sh 
```
```bash
bash ./scripts/ReProTS_ETTh2.sh 
```
```bash
bash ./scripts/ReProTS_ETTm1.sh 
```
```bash
bash ./scripts/ReProTS_ETTm2.sh
```

## Detailed usage

Please refer to ```run_main.py```, ```run_m4.py``` and ```run_pretrain.py``` for the detailed description of each hyperparameter.

