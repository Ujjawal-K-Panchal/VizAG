"""
Title: Test Generator.

Date: May 28, 2024; 4:47 PM

Author: Ujjawal K. Panchal & Ajinkya Chaudhari & Isha S. Joglekar
"""
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import peft

import projconfig

def get_tokenizer(
	name: str = projconfig.llm_name,
	model_max_length: int = projconfig.max_seq_len
):
	tok = AutoTokenizer.from_pretrained(
		name,
		cache_dir = projconfig.modelstore,
		model_max_length = model_max_length,
		token = projconfig.hf_token,
		quantization_config = projconfig.nf4_config
	)
	tok.padding_side = 'right'
	tok.model_max_length = max_seq_len
	tok.pad_token = tok.eos_token
	return tok

def get_model(
	name: str = projconfig.llm_name,
	quantize: Optional[str] = None,
	device = projconfig.device
):
	if quantize == "qlora":
		model = AutoModelForCausalLM.from_pretrained(
			name,
			quantization_config = projconfig.nf4_config,
			cache_dir = projconfig.modelstore,
			token = projconfig.hf_token,
		)
		model = peft.get_peft_model(model, projconfig.lora_config)
	else:
		model = AutoModelForCausalLM.from_pretrained(
			name,
			cache_dir = projconfig.modelstore,
			token = projconfig.hf_token,
		)
		if quantize == "half":
			model = model.to(torch.float16)
	model.to(device)
	model.eval()
	return model

def get_reply(model: AutoModelForCausalLM, tok: AutoTokenizer, msg: str):
	#1. format message.
	msg = {"role": "user", "content": msg}
	#2. tokenize message.
	input_ids = tokenizer.apply_chat_template(
		msg,
		add_generation_prompt=True,
		return_tensors="pt"
	).to(model.device)
	#3. generate with model.
	outputs = model.generate(
		input_ids,
		max_new_tokens=projconfig.max_seq_len,
		eos_token_id=tok.eos_token_id,
		do_sample=projconfig.do_sample,
		temperature=projconfig.temperature,
		top_p=projconfig.top_p,
	)
	reply = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
	return reply

if __name__ == "__main__":
	#1. get model & tokenizer.
	model, tok = get_model(), get_tokenizer()
	#2. get reply for a message.
	print(get_reply("Warren E. Buffett rocks!"))