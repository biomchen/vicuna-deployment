#### About 

# Vicuna-finetune-deployment

This is the rep of detailed steps about the deployment of the vicuna model and its finetuning steps.

### Vicuna model  
https://github.com/thisserand/FastChat

### GPTQ for LLaMA
https://github.com/oobabooga/GPTQ-for-LLaMa

[GPTQ](https://arxiv.org/abs/2210.17323) is the Post-Training Quantization for Generative Pre-trained Transformers, a SOTA one-shot weight quantization method.

#### Deply
```sh
python -m fastchat.serve.cli --model-name anon8231489123/vicuna-13b-GPTQ-4bit-128g --wbits 4 --groupsize 128
```
