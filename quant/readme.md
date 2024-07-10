

##要求
pip install imageio[ffmpeg] safetensors imageio beartype flash-attn

##int8 设置/相关更改
工具：torchao.quantization.quant_api.change_linear_weights_to_int8_dqtensors
量化模块：layers[i].mlp、layers[i].attention.query_key_value、layers[i].attention.dense
torch compile 设置： max-autotun及 torch._inductor.config相关setting

其他修改：在config.json 中加入了"is_distributed"、"precision" (默认int8模式下，不执行cp分布式)


##实际diff/infer time 
bf16 compiled runtime of the quantized block(int8) is 2712.94ms and peak memory  10.23GB
DIT Diff  tensor(0.0021, device='cuda:0', dtype=torch.bfloat16)



##运行
bash example.sh
