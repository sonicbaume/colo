# Colo Sum
- Inference wrapper for [COLO: A Contrastive Learning based Re-ranking Framework for
One-Stage Summarization](https://arxiv.org/pdf/2209.14569)
- Forked from [ordersum](github.com/Espresso-AI/ordersum)
## Dependencies
- python : 3.9
- torch : 1.13.1
- transformers : 4.34.0
- pytorch-lightning : 1.9.0
- rouge-score : 0.1.2

## Usage 
### Get sentences at 75% of original length
```bash
uv run predict.py -s sentences.json -r 0.75 -c /path/to/checkpoints.ckpt
```

### Get indexes at 75% of original length
```bash
uv run predict.py -s sentences.json -r 0.75 -c /path/to/checkpoints.ckpt -i
```

## License
BSD 3-Clause License Copyright (c) 2022