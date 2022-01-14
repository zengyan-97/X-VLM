from transformers.optimization import AdamW


def create_optimizer(args, model):
    lr = args.lr
    wd = args.weight_decay
    lr_mult = getattr(args, 'lr_mult', 1)
    print("### lr_mult, ", lr_mult)

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": wd, "lr": lr},
        {"params": [], "weight_decay": 0.0, "lr": lr},
        {"params": [], "weight_decay": wd, "lr": lr * lr_mult},
        {"params": [], "weight_decay": 0.0, "lr": lr * lr_mult}
    ]

    no_decay = {"bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight"}

    if hasattr(model, 'init_params'):
        large_lr = model.init_params
        print("### model has 'init_params', ", len(large_lr))
    else:
        large_lr = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights

        if any(nd in n for nd in no_decay):
            if n in large_lr:
                optimizer_grouped_parameters[3]['params'].append(p)
            else:
                optimizer_grouped_parameters[1]['params'].append(p)
        else:  # decay
            if n in large_lr:
                optimizer_grouped_parameters[2]['params'].append(p)
            else:
                optimizer_grouped_parameters[0]['params'].append(p)

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))

    return optimizer
