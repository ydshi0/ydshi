
from calflops import calculate_flops
from transformers import PreTrainedTokenizerFast
from transformers import LlamaForCausalLM
import sys
import re
import numpy as np
    
def TF(log_file,input_length):

    with open(log_file, "r", encoding="utf-8") as file:
        log_content = file.read()
    match = re.search(r"outputlen\s*\[([^\]]+)\]", log_content)
    outputlen_str = match.group(1)  # 提取方括号内部的内容
    outputlen = [int(num) for num in outputlen_str.split(",")]  # 转换为整数列表
    #print(outputlen)

    batch_size= 1
    
    model_save = "/host_model/meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = LlamaForCausalLM.from_pretrained(model_save)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_save)

    
    max_seq_length = input_length  
    ptflops, macs, params = calculate_flops(model=model,
                                        input_shape=(batch_size, max_seq_length),
                                        transformer_tokenizer=tokenizer,print_detailed=False)
    ptflops = float(ptflops.split()[0])

    max_seq_length = input_length + int(np.mean(outputlen))
    tmp, macs, params = calculate_flops(model=model,
                                        input_shape=(batch_size, max_seq_length),
                                        transformer_tokenizer=tokenizer,print_detailed=False)
    
    dtflops = float(tmp.split()[0]) - ptflops 

    num = len(outputlen)
    sum_prefill = num * ptflops
    sum_decode = num * dtflops

    #print(f"prefill-flops: {ptflops:.2f}, decode-flops: {dtflops:.2f} TFLOPS")

    return sum_prefill,sum_decode

if __name__ == "__main__":  # 只有直接运行该脚本时才会执行
    path = sys.argv[1]
    input_length = int(sys.argv[2])

    # 运行 FLOPs 计算
    TF(path, input_length)