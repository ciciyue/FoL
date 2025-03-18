# -*- coding: UTF-8 -*-
import torch


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    # If the checkpoint contains nested model state dictionary
    if 'model_state_dict' in checkpoint:
        # Load the nested model state dictionary
        model_state_dict = checkpoint['model_state_dict']
        # Remove "module." prefix from keys
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    else:
        # If not nested, assume model state dict is directly in the checkpoint
        model_state_dict = checkpoint
        # Remove "module." prefix from keys
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    # Load the model state dictionary
    model.load_state_dict(model_state_dict, strict=False)
    return model
