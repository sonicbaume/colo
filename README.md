# Colo Sum
- Inference wrapper for [COLO: A Contrastive Learning based Re-ranking Framework for
One-Stage Summarization](https://arxiv.org/pdf/2209.14569)
- Forked from [ordersum](github.com/Espresso-AI/ordersum)

## Install 
```bash
# if  testing on macos is required remove +cpu from the requirements.txt file
pip install -r requirements.txt
```

## Usage 
### Get sentences at 75% of original length
```bash
python3 predict.py -s sentences.json -r 0.75 -cc /path/to/checkpoint.ckpt
```

### Get indexes at 75% of original length
```bash
python3 predict.py -s sentences.json -r 0.75 -cc /path/to/colo-checkpoint.ckpt -bc /path/to/bart-checkpoint.ckpt -i
```

## License
BSD 3-Clause License Copyright (c) 2022