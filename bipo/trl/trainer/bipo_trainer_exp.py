from collections import defaultdict
from typing import Optional, override
import torch
from .bipo_trainer import BiPOTrainer

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False



class BiPOTrainerEXP(BiPOTrainer):
    def __init__(self, *args,pipeline:str = 'default', masking_type:str = 'soft', moving: Optional[str] = 'default', quantile: Optional[float] = 0.0, num_layer: Optional[int] = 26,  filter_step:int = 4,**kwargs):
        super().__init__(*args, **kwargs)
        self.fisher_accumulator = {}
        self.importance_map = {}
        self.quantile_threshold = quantile
        self.filter_step = filter_step
        self.pipeline = pipeline
        self.masking_type = masking_type
        self.moving = moving
        self.idx_param = {}

        self.idx_layer_tensor = torch.arange(num_layer, dtype=torch.float32)
        self.layer_weight = torch.arange(num_layer, dtype=torch.float32)

        # idx = 0
        for name, p in self.model.named_parameters():
            if "vec" in name:
                self.fisher_accumulator[name] = torch.zeros_like(p)
                # idx+=1

        print('[Pipeline:]',pipeline)
    

    # Router
    @override
    def training_step(self, model, inputs,num_items_in_batch=None):
        if self.pipeline == 'default':
            loss = super().training_step(model, inputs, num_items_in_batch)
        
        elif self.pipeline == 'one':
            loss = self._training_step_one(model, inputs,num_items_in_batch)
        
        elif self.pipeline == 'trace_analysis':
            loss = self._training_step_two(model, inputs,num_items_in_batch)

        return loss
        

        

    # EXP 9: Gradual Unfreezing with soft masking  
    def _training_step_one(self, model, inputs,num_items_in_batch=None):
        
        # Gradual Freezing
        self.quantile_threshold = getattr(self.state, "custom_quantile_threshold", self.quantile_threshold)
        
        if self.moving == 'backward':
            threshold = torch.quantile(self.idx_layer_tensor, 1-self.quantile_threshold) 
            hard_mask = self.idx_layer_tensor >= threshold
        else:
            threshold = torch.quantile(self.idx_layer_tensor, self.quantile_threshold) 
            hard_mask = self.idx_layer_tensor <= threshold
        
        vec_idx = 0
        for name, param in model.named_parameters():
            if "vec" in name:
                print(f"[Layer:] {vec_idx} Learning" if hard_mask[vec_idx].item() else f"[Layer:] {vec_idx} Freezing")
                param.requires_grad = hard_mask[vec_idx].item()
                vec_idx += 1

        loss = super().training_step(model, inputs,num_items_in_batch)

        with torch.no_grad():
            if self.state.global_step % self.filter_step == 0 and self.state.global_step > 0:
                for name, param in model.named_parameters():
                    if "vec" in name and param.grad is not None:
                        self.importance_map[name] = param.grad.pow(2)

            # if self.experiment_pipeline != 'one':
            #     for name, param in model.named_parameters():
            #         if name in self.importance_map and param.grad is not None:
            #             importance = self.importance_map[name].float()

            #             if self.masking_type == 'soft':
            #                 min_val = importance.min()
            #                 max_val = importance.max()
                            
            #                 soft_mask = (importance - min_val) / (max_val - min_val+ 1e-8)
            #                 param.grad.mul_(soft_mask)

            #             if self.masking_type == 'hard':
            #                 importance = self.importance_map[name].float()
            #                 threshold = torch.quantile(importance, 0.9)
            #                 hard_mask = (importance >= threshold).float()
                            
            #                 param.grad.mul_(hard_mask)

  

        return loss

     # EXP 11: Fisher Layer selection
    # @override
    # def training_step(self, model, inputs, num_items_in_batch=None):
    #     is_filter_step = (self.state.global_step % self.filter_step == 0)

    #     # 1. Initial mapping
    #     if self.state.global_step == 0:  
    #         vec_idx = 0
    #         for name, param in model.named_parameters():
    #             if "vec" in name:
    #                 self.idx_param[vec_idx] = param
    #                 vec_idx += 1

    #     if is_filter_step:
    #         for param in self.idx_param.values():
    #             param.requires_grad = True
                
    #     # 3. Forward and Backward pass
    #     loss = super().training_step(model, inputs, num_items_in_batch)

    #     if is_filter_step:
    #         with torch.no_grad():
    #             for idx, param in self.idx_param.items():
    #                 if param.grad is not None:
    #                     self.layer_weight[idx] = param.grad.pow(2).mean()

    #             self.quantile_threshold = getattr(self.state, "custom_quantile_threshold", self.quantile_threshold)
    #             threshold = torch.quantile(self.layer_weight, 1 - self.quantile_threshold) 
    #             hard_mask = self.layer_weight >= threshold

    #             hard_mask_list = hard_mask.tolist()
    #             for idx, param in self.idx_param.items():
    #                 is_learning = hard_mask_list[idx]
    #                 param.requires_grad = is_learning
                    
    #                 if not is_learning and param.grad is not None:
    #                     param.grad.zero_()

    #     return loss
    
    # https://arxiv.org/pdf/2503.11164
    # Layer sensitivity analysis using Fisher Information Matrix to Approximate Hessian matrix trace
    def _training_step_two(self,model, inputs,num_items_in_batch):

        if self.moving == 'backward' or self.moving =='forward':
            self.quantile_threshold = getattr(self.state, "custom_quantile_threshold", self.quantile_threshold)
            if self.moving == 'backward':
                threshold = torch.quantile(self.idx_layer_tensor, 1-self.quantile_threshold) 
                hard_mask = self.idx_layer_tensor >= threshold
            else:
                threshold = torch.quantile(self.idx_layer_tensor, self.quantile_threshold) 
                hard_mask = self.idx_layer_tensor <= threshold
        
            # vec_idx = 0
            for name, param in model.named_parameters():
                if "vec" in name or "gate_mask" in name:
                    parts = name.split('.')
                    layer_idx = int(parts[2])
                    param.requires_grad = hard_mask[layer_idx].item()
                    print(f"[Layer:] {layer_idx} {name} Learning" if hard_mask[layer_idx].item() else f"[Layer:] {layer_idx} Freezing")
                    # vec_idx += 1

        loss = super().training_step(model, inputs,num_items_in_batch)

        with torch.no_grad():
            if not hasattr(self, 'fisher_counter'):
                self.epoch_traces = {} 
                self.fisher_counter = defaultdict(int)
                self.last_recorded_epoch = 0
                
            idx = 0
            for name, param in model.named_parameters():
                if "vec" in name and param.grad is not None:
                    self.fisher_accumulator[name] += param.grad.to(torch.float32).pow(2)
                    self.fisher_counter[name] += 1
                    idx += 1
            

            current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0
            is_last_step = (self.state.global_step == self.state.max_steps - 1)

            if current_epoch > self.last_recorded_epoch or is_last_step:
                print(f"\n--- Saving Traces for Epoch {self.last_recorded_epoch} (Step {self.state.global_step}) ---")
                
                epoch_data = {}
                # traces_list = []
                for idx in self.fisher_accumulator:
                    if self.fisher_counter[idx] > 0:
                        avg_squared_grad = self.fisher_accumulator[idx] / self.fisher_counter[idx]
                        trace = avg_squared_grad.sum().item()
                        # traces_list.append(trace)
                    else:
                        trace = 0.0
                    
                    epoch_data[idx] = trace
                    print(f"Layer: {idx} | trace: {trace:.6f}")

                self.epoch_traces[self.last_recorded_epoch] = epoch_data

                for name in self.fisher_accumulator:
                    self.fisher_accumulator[name].zero_()
                    self.fisher_counter[name] = 0
                    
                self.last_recorded_epoch = current_epoch                

            if is_last_step:
                for epoch, traces in self.epoch_traces.items():
                    print(f"\n--- Traces for Epoch {epoch} ---")
                    for name, trace in traces.items():
                        print(f"Layer: {name} | trace: {trace:.6f}")                    

        return loss
        
    # def scale(traces, a=0.01, eps=1e-8):
    #     max_trace = torch.max(traces)
    #     trace_ratio = traces / (max_trace + eps)
    #     scale = a + (1.0 - a) * (1.0 - trace_ratio)
        
    #     return scale
