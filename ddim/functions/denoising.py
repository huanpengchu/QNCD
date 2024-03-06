import torch
from tqdm import tqdm
import copy 
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


# def generalized_steps(x, seq, model, b, **kwargs):
#     with torch.no_grad():
#         n = x.size(0)
#         seq_next = [-1] + list(seq[:-1])
#         x0_preds = []
#         xs = [x]
#         for i, j in zip(reversed(seq), reversed(seq_next)):
#             t = (torch.ones(n) * i).to(x.device)
#             next_t = (torch.ones(n) * j).to(x.device)
#             at = compute_alpha(b, t.long())
#             at_next = compute_alpha(b, next_t.long())
#             xt = xs[-1].to('cuda')
#             et = model(xt, t)
#             x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#             x0_preds.append(x0_t.to('cpu'))
#             c1 = (
#                 kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
#             )
#             c2 = ((1 - at_next) - c1 ** 2).sqrt()
#             xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
#             xs.append(xt_next.to('cpu'))

#     return xs, x0_preds

def generalized_steps(x, seq, model, b, do_calib, **kwargs):

    calib_img=None
    calib_t=None

    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        m = 0
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
            if(m%5==0 and do_calib):
                if(calib_img==None):
                    calib_img=copy.deepcopy(xt.detach().cpu())
                    calib_t=t.detach().cpu()
                else:
                    calib_img=torch.concat((calib_img,xt.detach().cpu()),dim=0)
                    calib_t=torch.concat((calib_t,t.detach().cpu()),dim=0)
            m=m+1

    return xs, x0_preds, (calib_img,calib_t)


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
