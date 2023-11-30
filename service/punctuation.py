import numpy as np
import transformers
import torch
import random
import os
import gc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
random.seed(2023)

class MachineCodeDetector:
    def __init__(self):
        print('*setting model*')
        self.cache_dir = './.cache'
        os.environ["XDG_CACHE_HOME"] = self.cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print(f"Using cache dir {self.cache_dir}")
        name = "codellama/CodeLlama-7b-hf"
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, cache_dir=self.cache_dir) #**base_model_kwargs,

        self.base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, cache_dir=self.cache_dir) #**optional_tok_kwargs,
        self.base_tokenizer.pad_token_id = self.base_tokenizer.eos_token_id
        

    async def generate(self, data):
        perturbation_results = await self.get_perturbation_results_fast(data, gamma=0, n_perturbations=1, n_samples=1)
        output = self.run_perturbation_experiment(perturbation_results)    
        return output
    
    async def get_discrepency(self, logits, labels):
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits, dim=-1)
        probs_ref = torch.softmax(logits, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()

    async def get_perturbation_results_fast(self, data, gamma=0.0, n_perturbations=1, n_samples=500):
        torch.manual_seed(0)
        np.random.seed(0)
        gc.collect()
        torch.cuda.empty_cache()
        
        text = data
        
        tokenized = self.base_tokenizer(text, truncation=True, return_tensors="pt").to(DEVICE)
        
        if tokenized.input_ids.nelement() > 2:
            original_logits = self.base_model(**tokenized).logits[:,:-1]
        else:
            result = {"original": text, "perturbed_original": None, "prediction_original": None}
            return result

        original_labels = tokenized.input_ids[:,1:]
        original_samples = torch.argmax(original_logits, dim=-1)
        perturbed_original= self.base_tokenizer.batch_decode(original_samples, skip_special_tokens=True)
        prediction_original = await self.get_discrepency(original_logits, original_labels)
            
        result = {"original": text, "perturbed_original": perturbed_original, "prediction_original": prediction_original}
        return result


    def run_perturbation_experiment(self, results):
        best_threshold = -0.925924
        if results['prediction_original'] == None:
            return "Try longer code"
        else:
            return "human" if results['prediction_original'] > best_threshold else "machine"