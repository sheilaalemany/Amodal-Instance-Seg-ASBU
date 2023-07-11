import numpy as np

import torch

def visualize_tensor(tensors_dict, mean, div):
    together = []

    for ct in tensors_dict['common_tensors']:
        ct = unormalize(ct.detach().cpu(), mean, div)
        ct *= 255
        ct = torch.clamp(ct, 0, 255)
        together.append(ct)

    for mt in tensors_dict['mask_tensors']:
        if mt.size(1) == 1:
            mt = mt.repeat(1,3,1,1)
        mt = mt.float().detach().cpu() * 255
        together.append(mt)
        
    # added by Sheila, we are trying to append the original images to the masks here
    if 'originals' in tensors_dict: 
        print('...we reached the point where we are appending the originals to together!')
        for ot in tensors_dict['originals']:
            ot = ot.detach().cpu()
            together.append(ot)

    part_tensor = tensors_dict.get('part_tensor', [])
    for pt in part_tensor:
        together.append(pt)        
        
    if len(together) == 0:
        return None
    together = torch.cat(together, dim=4) # changed from 3 to 4
    together = together.permute(1,0,2,3).contiguous()
    together = together.view(together.size(0), -1, together.size(4)) # changed from 3 to 4
    print('...we have successfully put everything together')
    return together

def unormalize(tensor, mean, div):
    for c, (m, d) in enumerate(zip(mean, div)):
        tensor[:,c,:,:].mul_(d).add_(m)
    return tensor
