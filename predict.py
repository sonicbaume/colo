from transformers import AutoTokenizer
from src.model.colo.bart_base import CoLo_BART
from src.model.utils import convert_checkpoints
from typing import List, Any
import torch
import argparse
import json
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLS_TOKEN = "<cls>"
SEP_TOKEN = "<sep>"
DOC_TOKEN = "<doc>" 

def predict_sentences(
    model: CoLo_BART, tokenizer: Any, sentences: List[str], keep_ratio: float
):
    if(keep_ratio >= 1):
        raise ValueError("keep_ratio", "should be less than 1: ", keep_ratio)
    
    out: list[str] = []
    chunks = split_sentences_into_chunks(
        sentences,
        tokenizer,
    )
    for chunk in chunks:
        # calculate the number of lines to keep
        num_sentences      = round(len(chunk) * keep_ratio)
        model.num_ext_sent =  num_sentences + 1  if num_sentences != len(chunk) else num_sentences # Top N sentences to keep for forming candidates
        model.num_can_sent = [num_sentences-1, num_sentences, num_sentences+1]                     # Try slightly shorter/longer summaries
        
        # tokenize sentences
        joined_text = f"{CLS_TOKEN} " + f" {CLS_TOKEN} ".join(chunk)
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
        encoding      = { k: v.to(DEVICE) for k, v in encoding.items() }
        cls_positions = cls_positions.to(DEVICE)
        
        # predict sentences
        with torch.no_grad():
            output = model(
                encodings=encoding,
                cls_token_ids=cls_positions,
            )

        top_ids = list(output['prediction'][0][0])
        top_ids.sort()
        out.extend([chunk[i] for i in top_ids])
        
    return out

def load_model_and_tokenizer(checkpoint: str):
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
            
    model.load_state_dict(convert_checkpoints(checkpoint, DEVICE))
    model.eval()
    model.to(DEVICE)
    
    return model, tokenizer
    
def split_sentences_into_chunks(sentences, tokenizer, max_tokens=1024, cls_token="<cls>"):
    chunks = []
    current_chunk = []
    current_token_count = 0

    for sentence in sentences:
        text_with_cls = f"{cls_token} {sentence}"
        token_count = len(tokenizer.encode(text_with_cls, add_special_tokens=False))

        # start new chunk if adding this sentence would exceed the max_tokens
        if current_token_count + token_count > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [sentence]
            current_token_count = token_count
        else:
            current_chunk.append(sentence)
            current_token_count += token_count

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def run(sentences: List[str], keep_ratio: float, checkpoint: str):
    try:         
        model, tokenizer = load_model_and_tokenizer(checkpoint)
        return predict_sentences(model, tokenizer, sentences, keep_ratio)
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
        metavar="CKPT",
        help="Path to model checkpoint (.ckpt)",
    )

    args = parser.parse_args(argv)

    if not (0.0 < args.ratio < 1.0):
        parser.error("The -r / --ratio value must be between 0 and 1.")

    return args

def main(argv=None):
    args = parse_cli(argv)
    results = run(args.sentences, args.ratio, args.checkpoint)
    # todo format to JSON
    json.dump(results, sys.stdout, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()