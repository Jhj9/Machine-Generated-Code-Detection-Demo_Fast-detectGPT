import numpy as np
import transformers
import torch
import random
from sklearn.metrics import precision_recall_curve, auc, classification_report
import argparse
import os
import gc

def get_discrepency(logits, labels):
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits, dim=-1)
    probs_ref = torch.softmax(logits, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def get_precision_recall_metrics(real_preds, sample_preds,best_threshold=0):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    preds = [0 if i <= best_threshold else 1 for i in real_preds+sample_preds]
    print(classification_report([0] * len(real_preds) + [1] * len(sample_preds), preds))
    return precision.tolist(), recall.tolist(), float(pr_auc)

def get_perturbation_results_fast(gamma=0.0, n_perturbations=1, n_samples=500):
    torch.manual_seed(0)
    np.random.seed(0)
    gc.collect()
    torch.cuda.empty_cache()
    
    text = data['code']
    
    tokenized = base_tokenizer(text, truncation=True, return_tensors="pt").to(DEVICE)
    
    if tokenized.input_ids.nelement() > 2:
        original_logits = base_model(**tokenized).logits[:,:-1]
    else:
        return [{"original": text, "perturbed_original": None, "prediction_original": None}]

    original_labels = tokenized.input_ids[:,1:]
    original_samples = torch.argmax(original_logits, dim=-1)
    perturbed_original=base_tokenizer.batch_decode(original_samples, skip_special_tokens=True)
    prediction_original = get_discrepency(original_logits, original_labels)
        
    return {"original": text, "perturbed_original": perturbed_original, "prediction_original": prediction_original}


def run_perturbation_experiment(results, criterion, span_length=10, n_perturbations=1, n_samples=500):
    best_threshold = 0
    if results['prediction_original'] == None:
        return "Try longer code"
    else:
        return "human" if results['prediction_original'] > best_threshold else "machine"

def load_base_model_and_tokenizer(name):
    if args.openai_model is None:
        print(f'Loading BASE model {args.base_model_name}...')
        base_model_kwargs = {}
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None
    
    optional_tok_kwargs = {}
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_model, base_tokenizer

if __name__ == '__main__':
    DEVICE = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)
    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="./.cache")
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--code', type=str, default="")
    args = parser.parse_args()

    #API_TOKEN_COUNTER = 0

    #precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    #sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    #output_subfolder = f"{args.output_name}/" if args.output_name else ""
    if args.openai_model is None:
        base_model_name = args.base_model_name.replace('/', '_')
    #else:
    #    base_model_name = "openai-" + args.openai_model.replace('/', '_')
    scoring_model_string = (f"-{args.scoring_model_name}" if args.scoring_model_name else "").replace('/', '_')
    
    #mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    #n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    #n_perturbation_rounds = args.n_perturbation_rounds
    #n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # generic generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(2023)


    data = {'code': args.code}
    
    perturbation_results = get_perturbation_results_fast(args.gamma, 1, n_samples)
    #for perturbation_mode in ['d', 'z']:
    perturbation_mode = "fast"
    output = run_perturbation_experiment(
        perturbation_results, perturbation_mode, span_length=args.span_length, n_perturbations=1, n_samples=n_samples)
    #outputs.append(output)
    print(output,"!!!")