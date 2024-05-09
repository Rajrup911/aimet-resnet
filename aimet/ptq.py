import torch
from tqdm import tqdm
import timm
import numpy as np
import os

import onnxruntime as ort
import onnx

from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.model_preparer import prepare_model

class aimetPTQ:
    def __init__(self, config) -> None:
        self.config = config
        self.model = self.get_model()

    def get_model(self):
        model = timm.create_model(self.config.model.call_name,
                                  self.config.model.pretrained,
                                  num_classes = self.config.model.num_classes).cuda()
                                  
        dummy_input = torch.rand(1, *self.config.model.input_shape).cuda()
        torch.onnx.export(model, dummy_input, os.path.join(self.config.model.artifacts, "resnet_fp32.onnx"), export_params=True, opset_version=13, do_constant_folding=True)
        return model
    
    def dataloader(self, model):
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training = False)
        val_dataset = timm.data.ImageDataset(self.config.model.dataset_dir, transform = transforms)
        val_dataset_sub = torch.utils.data.Subset(val_dataset, list(range(1000)))
        val_loader = timm.data.create_loader(val_dataset, (1, *self.config.model.input_shape), 1)
        val_loader_sub = timm.data.create_loader(val_dataset_sub, (1, *self.config.model.input_shape), 1)
        return val_loader,val_loader_sub
    
    def validate(self, model, val_loader):
        correct = 0
        total = 0
        elapsed_time=0
        top5_correct=0
        
        with torch.inference_mode():
            for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc="Validation Progress")):
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs.softmax(dim=1) * 100, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
                _, top5_predicted = torch.topk(outputs.softmax(dim=1) * 100,k=5)
                top5_correct += sum(labels[i].item() in top5_predicted[i] for i in range(len(labels)))
        
        top1Accuracy = correct / total
        top5Accuracy = top5_correct / total
        print('Top 1 Accuracy : {:.2f}%\nTop 5 Accuracy : {:.2f}%\n'.format(100 * top1Accuracy,100 * top5Accuracy))
        
    def prepare_model(self, model):
        return prepare_model(model)
    
    def cross_layer_equalization_auto(self, model):
        input_shape = (1, *self.config.model.input_shape)
        
        dummy_input = torch.rand(1, *self.config.model.input_shape).cuda()
        #model = self.prepare_model(model)
        equalize_model(model.cuda(), input_shape)
        torch.onnx.export(model, dummy_input, os.path.join(self.config.model.artifacts, "resnet_after_CLE.onnx"), export_params=True, opset_version=13, do_constant_folding=True)
        #_ = fold_all_batch_norms(model, input_shapes = input_shape)
    
    def make_sim(self, model):
    
        use_cuda = False
        if torch.cuda.is_available():
            use_cuda = True
            
        sim = QuantizationSimModel(model.cuda(), quant_scheme = QuantScheme.post_training_tf_enhanced, default_output_bw=8, default_param_bw=8, 
                                   dummy_input = torch.rand(1, *self.config.model.input_shape).cuda())
                                   
        sim.set_and_freeze_param_encodings(encoding_path=os.path.join(self.config.model.artifacts, 'adaround.encodings'))
        sim.compute_encodings(forward_pass_callback = self.pass_calibration_data, forward_pass_callback_args = use_cuda)
        
        dummy_input = torch.rand(1, *self.config.model.input_shape).cpu()
        sim.export(path = self.config.model.artifacts, filename_prefix='resnet18_adaround', dummy_input=dummy_input)
        return sim
        
    def adaround(self, model):
        val_dataset = timm.data.ImageDataset(self.config.model.dataset_dir)
        val_loader = timm.data.create_loader(val_dataset, (1, *self.config.model.input_shape), 32)
        
        params = AdaroundParameters(data_loader = val_loader, num_batches = 32, default_num_iterations = 10000)
    
        dummy_input = torch.rand(1, *self.config.model.input_shape).cuda()
        ada_model = Adaround.apply_adaround(model.cuda(), dummy_input, params,
                                            path = self.config.model.artifacts,
                                            filename_prefix = 'adaround',
                                            default_param_bw = 8,
                                            default_quant_scheme = QuantScheme.post_training_tf_enhanced)
        return ada_model
        
    def pass_calibration_data(self, sim_model, use_cuda):
    
        val_dataset = timm.data.ImageDataset(self.config.model.dataset_dir)
        val_loader = timm.data.create_loader(val_dataset, (1, *self.config.model.input_shape), 1)
    
        batch_size = 1
        max_batch_counter = 100
    
        sim_model.eval()
        if use_cuda:
            device = torch.device('cuda')
    
        current_batch_counter = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
    
                inputs_batch = images.to(device)
                sim_model(inputs_batch)
    
                current_batch_counter += 1
                if current_batch_counter == max_batch_counter:
                    break
