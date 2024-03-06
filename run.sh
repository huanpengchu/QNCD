export CUDA_VISIBLE_DEVICES=2
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" \
--plms --cond  --weight_bit 4 --quant_mode qdiff --quant_act --act_bit 8 \
--cali_st 25 --cali_batch_size 1 --cali_n 128 --no_grad_ckpt --split --ptq \
--resume_w --cali_ckpt w4_ours.pth \
--scale_split_static --sample_reverse --reverse_interval 10 \
--running_stat --sm_abit 16 --from_file val_ann.txt --n_samples 2 --outdir ckpts/ours_w4a8
