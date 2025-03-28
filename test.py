
from calflops import calculate_flops
from transformers import PreTrainedTokenizerFast
from transformers import LlamaForCausalLM
import sys
import re
import numpy as np
from CY_tflops import TF_main

def fun():

   


    
    prefill_128,decode = TF_main(128,1,1) 
    prefill_128 /= 1e12
    decode /= 1e12  
    print(f"CY p128-d1 {decode:.6f} TFLOPS")

    prefill_129,decode = TF_main(129,1,1) 
    prefill_129 /= 1e12
    
    decode_dis = prefill_129 - prefill_128

    print(f"CY p129-p128 {decode_dis:.6f} TFLOPS")

    prefill_512,decode = TF_main(512,1,1) 
    prefill_512 /= 1e12

    prefill_513,decode = TF_main(513,1,1) 
    prefill_513 /= 1e12

    decode_dis = prefill_513 - prefill_512

    print(f"CY p513-p512 {decode_dis:.6f} TFLOPS")

fun()


'''

    batch_size= 1
    
    model_save = "/host_model/meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = LlamaForCausalLM.from_pretrained(model_save)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_save)

    p1024, macs, params = calculate_flops(model=model,
                                input_shape=(batch_size, 1024),
                                transformer_tokenizer=tokenizer,print_detailed=False, print_results=False)
    p1024 = float(p1024.split()[0])

    p1025, macs, params = calculate_flops(model=model,
                                input_shape=(batch_size, 1025),
                                transformer_tokenizer=tokenizer,print_detailed=False, print_results=False)
    p1025 = float(p1025.split()[0])
    dtflops = p1025 - p1024
    print(f"p1024-flops: {p1024:.6f}, d-1-flops: {dtflops:.6f} TFLOPS")

    prefill,decode = TF_main(1024,1,1) 
    prefill /= 1e12
    decode /= 1e12  
    print(f"CY p1024-flops: {prefill:.6f}, d-1-flops: {decode:.6f} TFLOPS")

    p512, macs, params = calculate_flops(model=model,
                                        input_shape=(batch_size, 512),
                                        transformer_tokenizer=tokenizer,print_detailed=False, print_results=False)
    p512 = float(p512.split()[0])

    p513, macs, params = calculate_flops(model=model,
                                        input_shape=(batch_size, 513),
                                        transformer_tokenizer=tokenizer,print_detailed=False, print_results=False)
    p513 = float(p513.split()[0])
    dtflops = p513 - p512 
    print(f"p512-flops: {p512:.6f}, d-1-flops: {dtflops:.6f} TFLOPS")

    prefill,decode = TF_main(512,1,1) 
    prefill /= 1e12
    decode /= 1e12  
    print(f"CY p512-flops: {prefill:.6f}, d-1-flops: {decode:.6f} TFLOPS")


    p256, macs, params = calculate_flops(model=model,
                                    input_shape=(batch_size, 256),
                                    transformer_tokenizer=tokenizer,print_detailed=False, print_results=False)
    p256 = float(p256.split()[0])

    p257, macs, params = calculate_flops(model=model,
                                input_shape=(batch_size, 257),
                                transformer_tokenizer=tokenizer,print_detailed=False, print_results=False)
    p257 = float(p257.split()[0])
    dtflops = p257 - p256
    print(f"p256-flops: {p256:.6f}, d-1-flops: {dtflops:.6f} TFLOPS")

    prefill,decode = TF_main(256,1,1) 
    prefill /= 1e12
    decode /= 1e12  
    print(f"CY p256-flops: {prefill:.6f}, d-1-flops: {decode:.6f} TFLOPS")

    p128, macs, params = calculate_flops(model=model,
                                input_shape=(batch_size, 128),
                                transformer_tokenizer=tokenizer,print_detailed=False, print_results=False)
    p128 = float(p128.split()[0])

    p129, macs, params = calculate_flops(model=model,
                                input_shape=(batch_size, 129),
                                transformer_tokenizer=tokenizer,print_detailed=False, print_results=False)
    p129 = float(p129.split()[0])
    dtflops = p129 - p128

    print(f"p128-flops: {p128:.6f}, d-1-flops: {dtflops:.6f} TFLOPS")
'''