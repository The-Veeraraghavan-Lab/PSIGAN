
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'PSIGAN':
        assert(opt.dataset_mode == 'unaligned')
        from .PSIGAN_model import PSIGAN
        model = PSIGAN()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    
    print (opt.model)
    print (model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
