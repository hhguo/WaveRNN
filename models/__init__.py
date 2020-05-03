from hparams import hparams

def get_model(hparams):
    if hparams.mode == 'SG':
        from models.sg_wavernn import Model
        return Model
    else:
        from models.fatchord_wavernn import Model
        return Model