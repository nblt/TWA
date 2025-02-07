from torch.optim.optimizer import Optimizer, required
import copy
from copy import deepcopy
import torch
import numpy as np
import torch.optim._functional as F


class EntropyAdam(Optimizer):
    def __init__(self, params, config = {}):

        defaults = dict(lr=0.01, gtime=1,betas=(0.9, 0.999),eps=1e-8,
                        momentum_sgld=0, damp=0,
                        weight_decay_sgld=1e-4, weight_decay=0, nesterov=True,
                        L=0, eps_sgld=1e-4, g0=None, g1=None, gmax=1e4,
                        sgld_lr=0.1, alpha_arg=0.75, gscale=True, amsgrad=False)
        for k in defaults:
            if config.get(k, None) is None:
                config[k] = defaults[k]

        super(EntropyAdam, self).__init__(params, config)

    def __setgroup__(self, group):
        super(EntropyAdam, self).__setgroup__(group)

    @torch.no_grad()
    def step(self, closure=None):
        assert (closure is not None), \
                'attach closure for Entropy-SGD, model and criterion'

        group = self.param_groups[0]
        mom_sgld = group['momentum_sgld']
        mom_wd = group['weight_decay_sgld']
        wd = group['weight_decay']
        damp = group['damp']
        nesterov = group['nesterov']
        L = int(group['L'])
        eps_sgld = group['eps_sgld']
        gmax = group['gmax']
        
        sgld_lr = group['sgld_lr']
        alpha_arg = group['alpha_arg']
        gscale = group['gscale']
        
        gtime = group['gtime']

        # initialize
        params = group['params']
        if 'step' not in group:
            group['step'] = 0
            group['wc'], group['mdw'] = [], []

            for w in params:
                group['wc'].append(deepcopy(w.data))
                group['mdw'].append(deepcopy(w.data))

            # momentum init.
            for i, w in enumerate(params):
                group['mdw'][i].zero_()
                
            group['langevin'] = dict(mw=deepcopy(group['wc']),
                                     mdw=deepcopy(group['mdw']),
                                     eta=deepcopy(group['mdw']),
                                     lr_in=sgld_lr,
                                     alpha=alpha_arg)

        # SGLD init.
        lp = group['langevin']
        for i, w in enumerate(params):
            group['wc'][i].copy_(w.data)
            lp['mw'][i].copy_(w.data)
            lp['mdw'][i].zero_()
            lp['eta'][i].normal_()

            
        llr, alpha = lp['lr_in'], lp['alpha']
        
        g = group['g0'] * (1 + group['g1']) ** group['step']
                
        # SGLD loop
        for i in range(L):
            with torch.enable_grad():         
                mf = closure()
            
            # g scoping
            g = group['g0'] * (1 + group['g1']) ** group['step']
            g = min(g, gmax)
    
            for wc, w, mw, mdw, eta in zip(group['wc'], params, lp['mw'], lp['mdw'], lp['eta']):
                
                dw = w.grad.data
                
                # add interaction term
                dw.add_(wc - w.data, alpha=-g)

                # momentum and weight decay
                if mom_wd > 0:
                    dw.add_(w.data, alpha=mom_wd)
                
                if mom_sgld > 0:
                    mdw.mul_(mom_sgld).add_(dw, alpha=1-damp)
                    if nesterov:
                        dw.add_(mdw, alpha=mom_sgld)
                    else:
                        dw = mdw

                # add noise
                if eps_sgld > 0.:
                    eta.normal_()
                    dw.add_(eta, alpha=eps_sgld/np.sqrt(0.5*llr))

                # update weights
                w.data.add_(dw, alpha=-llr)
                mw.mul_(alpha).add_(w.data,alpha=1-alpha)

            # calculate g0 and g1 automatically (after 1 epoch)
            if group['step'] >= gtime:
                if group['g1'] == 0:
                    group['g1'] = group['gmax']**(1/(epochs*num_batches)) - 1
                if group['g0'] == 0 and i == L-1:
                    with torch.no_grad():
                        dist_0 = 0.
                        for w1, w2 in zip(group['wc'], params):
                            dist_0 += torch.sum((w1.data - w2.data)**2)
                    group['g0'] = mf.item() / (0.5*dist_0.item())
                    print(f"# COUPLING SCHEDULE  dist at step {group['step']}: {dist_0} g0: {group['g0']}  grate: {group['g1']}")

        # copy model back
        if L > 0:
            for i, w in enumerate(params):
                w.data.copy_(group['wc'][i])
                w.grad.data.copy_(w.data - lp['mw'][i])            
            
        # update parameters
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            if gscale:
                F.adam(params_with_grad,
                       grads,
                       exp_avgs,
                       exp_avg_sqs,
                       max_exp_avg_sqs,
                       state_steps,
                       amsgrad=group['amsgrad'],
                       beta1=beta1,
                       beta2=beta2,
                       lr=group['lr']*g,
                       weight_decay=group['weight_decay'],
                       eps=group['eps'])
            else:
                F.adam(params_with_grad,
                       grads,
                       exp_avgs,
                       exp_avg_sqs,
                       max_exp_avg_sqs,
                       state_steps,
                       amsgrad=group['amsgrad'],
                       beta1=beta1,
                       beta2=beta2,
                       lr=group['lr'],
                       weight_decay=group['weight_decay'],
                       eps=group['eps'])

                   
        return g


if __name__ == "__main__":


    x = torch.randn(10,10)
    x.requires_grad = True
    y = torch.randn(10,10)

    criterion = torch.nn.MSELoss()
    loss = criterion(x, y)
    loss.backward()
    print(loss)

    optimizer = EntropyAdam([x])

    optimizer.step()

    



