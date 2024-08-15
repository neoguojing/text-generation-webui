# git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git
# cd GPTQ-for-LLaMa
# pip install -r requirements.txt

# Quantize weights into INT4 and save as safetensors
# Quantized weight with parameter "--act-order" is not supported in TRT-LLM
python llama.py /data/model/llama3 c4 --wbits 4 --true-sequential --groupsize 128 --save_safetensors /data/model/llama3/llama-4bit-gs128.safetensors

python convert_checkpoint.py --model_dir /tmp/llama-7b-hf \
                             --output_dir ./tllm_checkpoint_gptq \
                             --dtype float16 \
                             --quant_ckpt_path /data/model/llama3/llama-4bit-gs128.safetensors  \
                             --use_weight_only \
                             --weight_only_precision int4_gptq \
                             --per_group 


trtllm-build --checkpoint_dir /data/model/llama3/tllm_checkpoint_gptq \
            --output_dir /data/model/llama3/trt_engines/int4_GPTQ/ \
            --gemm_plugin auto