import os


class Paths :
    def __init__(self, data_path, model_id) :
        self.data = data_path
        self.quant = '{}/quant/'.format(data_path)
        self.mel = '{}/mel/'.format(data_path)
        self.checkpoints = 'checkpoints/{}/'.format(model_id)
        self.latest_weights = '{}latest_weights.pyt'.format(self.checkpoints)
        self.output = 'model_outputs/{}/'.format(model_id)
        self.step = '{}/step.npy'.format(self.checkpoints)
        self.log = '{}log.txt'.format(self.checkpoints)
        self.create_paths()

    def create_paths(self) :
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.output, exist_ok=True)
