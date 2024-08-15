# emo-abc-gen
[![Python application](https://github.com/monet-joe/emo-abc-gen/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/monet-joe/emo-abc-gen/actions/workflows/python-app.yml)

Emotionally Conditioned Music Generation in ABC Notation with Chord

![image](https://github.com/monet-joe/emo-abc-gen/assets/20459298/9ee364d5-f80f-460d-9154-58b85ad59d15)

Referring to the RLBH of InstructGPT, we introduce a PPO reinforcement learning fine-tuning optimization for the tunesformer model as well.

## Requirements
```bash
conda create -n gpt --yes --file conda.txt
conda activate gpt
pip install -r requirements.txt
```

## Maintenance
```bash
git clone git@gitee.com:MuGeminorum/emo-abc-gen.git
cd emo-abc-gen
```

## Success rate
| Dataset |   %   |
| :-----: | :---: |
| Rough4Q |  100  |
| EMOPIA  |  24   |
