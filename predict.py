from transformers import AutoTokenizer
from src.model.colo.bart_base import CoLo_BART
from src.model.utils import convert_checkpoints
from typing import List, Any, Union
import torch
import argparse
import json
from pathlib import Path

CLS_TOKEN = "<cls>"
SEP_TOKEN = "<sep>"
DOC_TOKEN = "<doc>" 

def predict_sentences(
    model: CoLo_BART, tokenizer: Any, sentences: List[str], keep_ratio: float, device='cpu'
):
    if(keep_ratio >= 1):
        raise ValueError("keep_ratio", "should be less than 1: ", keep_ratio)
    
    # calculate the number of lines to keep
    num_sentences      = round(len(sentences) * keep_ratio)
    model.num_ext_sent =  num_sentences + 1  if num_sentences != len(sentences) else num_sentences # Top N sentences to keep for forming candidates
    model.num_can_sent = [num_sentences-1, num_sentences, num_sentences+1]                         # Try slightly shorter/longer summaries
    
    # tokenize sentences
    joined_text = f"{CLS_TOKEN} " + f" {CLS_TOKEN} ".join(sentences)
    encoding    = tokenizer(
        joined_text, 
        return_tensors='pt',
        padding='max_length', 
        truncation=True, 
        max_length=1024
    )
    input_ids     = encoding['input_ids'][0]
    cls_token_id  = tokenizer.convert_tokens_to_ids(CLS_TOKEN)
    cls_positions = (input_ids == cls_token_id).nonzero(as_tuple=True)[0].unsqueeze(0)
    encoding      = { k: v.to(device) for k, v in encoding.items() }
    cls_positions = cls_positions.to(device)
    
    # predict sentences
    with torch.no_grad():
        output = model(
            encodings=encoding,
            cls_token_ids=cls_positions,
        )
    
    top_ids: list[int] = list(output['prediction'][0][0])
    top_ids.sort()
    
    return top_ids

def load_model_and_tokenizer(checkpoint: Path, device = 'cpu'):
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
    new_tokens = [CLS_TOKEN, SEP_TOKEN, DOC_TOKEN]

    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    
    model = CoLo_BART(
        tokenizer,
        base_checkpoint  = "facebook/bart-large-cnn",
        enc_num_layers   = 0,
        enc_dropout_prob = 0.1,
        margin           = 0.01,
        alpha            = 1.0,
        beta             = 1.0
    )
            
    model.load_state_dict(convert_checkpoints(checkpoint, device))
    model.eval()
    model.to(device)
    
    return model, tokenizer
    
def count_tokens(sentences, tokenizer):
    total_tokens = 0
    
    for sentence in sentences:
        text_with_cls = f"{CLS_TOKEN} {sentence}"
        sentence_tokens = len(tokenizer.encode(text_with_cls, add_special_tokens=False))
        total_tokens += sentence_tokens
        
    return total_tokens

def run(sentences_path: Path, keep_ratio: float, checkpoint: Path, return_indexes):
    try:
        sentences: list[str] = json.load(open(sentences_path, "r"))    
        model, tokenizer = load_model_and_tokenizer(checkpoint)
        if(keep_ratio >= 1):
            raise ValueError("keep_ratio", "should be less than 1: ", keep_ratio)
        
        numInputToken = count_tokens(sentences, tokenizer)
        maxTokens     = 1024
        
        if(numInputToken > maxTokens): 
            raise ValueError(f'Input token count ({numInputToken}) exceeds ({maxTokens})')
        

        top_ids = predict_sentences(model, tokenizer, sentences, keep_ratio)
        
        if(return_indexes):
            return top_ids
        else:
            return [sentences[i] for i in top_ids]
            
        
    except BaseException as e:
        print(f"[error]: {e}")
        return None

def parse_cli(argv=None):
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Defaults to `sys.argv[1:]`.  Pass an explicit list for unit testing.

    Returns
    -------
    Namespace
        Parsed arguments with attributes: sentences, ratio, checkpoint.
    """
    parser = argparse.ArgumentParser(
        description="Run sentence-level predictions with a trained checkpoint."
    )
    parser.add_argument(
        "-s",
        "--sentences",
        required=True,
        type=Path, 
        metavar="PATH",
        help="Path to sentences JSON file",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        required=True,
        type=float,
        metavar="FLOAT",
        help="Threshold/ratio in the range 0-1",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        type=Path,
        metavar="CKPT",
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "-i", "--indexes",
        action="store_true",
        default=False,                
        help="Return the indices of the kept lines."
    )
    args = parser.parse_args(argv)

    if not (0.0 < args.ratio < 1.0):
        parser.error("The ratio value must be between 0 and 1.")

    return args

def main(argv=None):
    args = parse_cli(argv)
    results = run(args.sentences, args.ratio, args.checkpoint, args.indexes)
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()