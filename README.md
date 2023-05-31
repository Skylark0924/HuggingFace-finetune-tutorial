# HuggingFace-finetune-tutorial

In this tutorial, we summarized all pre-trained models supported by HuggingFace, and provided the finetuning code for 
each model. The models are divided into 4 task categories:
- Natural Language Processing (NLP)
- Computer Vision (CV)
- Audio Processing (AP)
- Multi-Modal (MModal)

## Install 

```
git clone https://github.com/Skylark0924/HuggingFace-finetune-tutorial
cd HuggingFace-finetune-tutorial
pip install -r requirements.txt 
```

## Usage

### Sign up HuggingFace and add your API token

1. Sign up HuggingFace: https://huggingface.co/join
2. Create and copy your API token: https://huggingface.co/settings/tokens
3. Add your API token to `token.txt` under the root directory of this repository

### Fine-tune a model

Fine-tune a model is very easy, just run the file related to your interested task. 
For example, if you want to fine-tune a model for text classification, just run the following command:

```
cd NLP
python text_classification_finetune.py
```

> **Note**
> If you want to fine-tune a model on your own dataset, please modify the `dataset` variable in the file. And you 
> can also use specific model by modifying the `pre_trained_model` variable.
