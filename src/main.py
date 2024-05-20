from benedict import benedict
import argparse
import json
import torch

from aimet import aimetPTQ

class JsonHandler:
    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            return benedict(data)

def main(config: benedict):

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        
    worker = aimetPTQ(config)    
    model = worker.get_model().cuda()
    
    val_loader, val_loader_sub = worker.dataloader(model)
    
    worker.validate(model.eval(), val_loader_sub)
    
    worker.cross_layer_equalization_auto(model)
    model = worker.adaround(model)
    sim = worker.make_sim(model)
    
    worker.validate(sim.model, val_loader_sub)
    
if __name__ == "__main__":
    def get_argparse():
        parser = argparse.ArgumentParser(description='Create')
        parser.add_argument('--config', required=True,  type=str,   help = "")
        args = parser.parse_args()
        return args

    args = get_argparse()
    cfg = benedict(JsonHandler.load_json(args.config))
    cfg.args = vars(args)
    main(cfg)