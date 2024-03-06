import torch.distributed as dist
import torch
# from qdiff.layer_recon import layer_reconstruction
# from qdiff.block_recon import block_reconstruction
from qdiff.quant_model import QuantModel
import logging
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
# logger = logging.getLogger(__name__)

def set_emb_float(model):
    for name,module in model.named_modules():
        if('emb' in name):
            try:
                print(name)
                module.set_quant_state(False,False)
                module.ignore_reconstruction=True
            except:
                continue

def run_once(qnn,cali_data,opt):
    cali_xs, cali_ts, cali_cs = cali_data
    if(len(cali_xs)==1):
        qnn(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
        qnn.set_running_stat(False, opt.rs_sm_only)
        return 

    calib_batch_size=2
    inds = np.random.choice(cali_xs.shape[0], calib_batch_size, replace=False)
    _ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda(), cali_cs[inds].cuda())
    
    inds = np.arange(cali_xs.shape[0])
    qnn.set_running_stat(True, opt.rs_sm_only)
    for i in trange(int(cali_xs.size(0) / calib_batch_size)):
        _ = qnn(cali_xs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(), 
            cali_ts[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
            cali_cs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda())
    qnn.set_running_stat(False, opt.rs_sm_only)


def calculate_scale(qnn,data,opt):
    if(not opt.scale_split_static):
        return 
    def set_scale_state(qnn,scale_split_static,scale_getting,set_weight):
        #pdb.set_trace()
        for name,module in qnn.named_modules():
            #print(name)
            if(hasattr(module,'scale_split_static')):
                print(module.use_scale_shift_norm,name)
                module.scale_split_static=scale_split_static
                module.scale_getting=scale_getting
                if(set_weight):
                    # import pdb
                    # pdb.set_trace()
                    module.set_conv_weight()

    qnn.set_quant_state(False,False)
    set_scale_state(qnn,opt.scale_split_static,True,False)
    run_once(qnn,data,opt)
    set_scale_state(qnn,opt.scale_split_static,False,True)

    #model.model.diffusion_model = qnn
    # classes=list(range(1000))
    # run_once(model, torch.device(f'cuda:0'), classes[0::40], 4,opt)



def resume_cali_model_brecq_multi(gpu, model, wq_params, aq_params, cali_data, quant_act, act_quant_mode,opt,outpath,logger):
    print(f"set_weight_quantize_params// brecq cali_model")
    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))
    rank = gpu
    

    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:12345',
                            world_size=4,rank=rank)
    torch.cuda.set_device(gpu)
    model.cuda()
    model.eval()
    qnn = QuantModel(
            model=model, weight_quant_params=wq_params, act_quant_params=aq_params, 
            sm_abit=8)
    qnn.cuda()
    qnn.eval()
    all_lists = []
    # import ipdb
    # ipdb.set_trace()
    # print(qnn)
    def recon_model(model):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
            if name == 'output_blocks':
                logger.info("Finished calibrating input and mid blocks, saving temporary checkpoint...")
                in_recon_done = True
                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
            if name.isdigit() and int(name) >= 9:
                logger.info(f"Saving temporary checkpoint at {name}...")
                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)
            torch.cuda.empty_cache()
    
    if(opt.scale_split_static):
        qnn.set_quant_state(False, False)
        calculate_scale(qnn,cali_data,opt)
    qnn.set_quant_state(True, False)
    if(opt.emb_float):
        set_emb_float(qnn)
    from qdiff.quant_block import QuantBasicTransformerBlock
    ig_state=True
    for name,module in qnn.named_modules():
        # if(ig_state==True):
        #     module.ignore_reconstruction = True
        if isinstance(module,QuantBasicTransformerBlock):
            print(name)
            module.ignore_reconstruction = True
            module.set_quant_state(False, False)
            ig_state=False

    num_cur_samples = cali_data[0].shape[0] // 4
    cur_cali_data = []
    cali_xs, cali_ts, cali_cs = cali_data
    _ = qnn(cali_xs[:2].cuda(), cali_ts[:2].cuda(), cali_cs[:2].cuda())

    for i in range(len(cali_data)):
        cur_cali_data.append(cali_data[i][gpu * num_cur_samples : (gpu+1) * num_cur_samples])
    cur_cali_data = tuple(cur_cali_data)

    # kwargs = dict(
    # cali_data=cur_cali_data, batch_size=opt.cali_batch_size, iters=opt.cali_iters_a, act_quant=True, 
    # opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p, cond=opt.cond,multi_gpu=True)

    kwargs = dict(cali_data=cur_cali_data, batch_size=opt.cali_batch_size, 
                iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                warmup=0.2, act_quant=False, opt_mode='mse', cond=opt.cond,multi_gpu=True)
    

    recon_model(qnn)
    torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))


    