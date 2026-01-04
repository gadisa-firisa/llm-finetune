import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class ChatModel:
    def __init__(self, base: str, adapter: str | None = None):
        self.tok = AutoTokenizer.from_pretrained(base, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, messages, max_new_tokens=256, temperature=0.2, top_p=0.95):
        inputs = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        input_len = inputs.shape[-1]
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature is not None and temperature > 0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tok.eos_token_id,
            )
        gen_ids = out[0][input_len:]
        return self.tok.decode(gen_ids, skip_special_tokens=True).strip()

