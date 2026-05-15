import torch
import torch.nn as nn

class KnowledgeExtractor(object):
    def __init__(self):
        self.knowledge_attn = {} 
        self.knowledge_emb = {}

    def register_efm(self, model, model_prefix, detach=False):
        for name, module in model.named_modules():
            if "inter_window_attn" in name:
                key = f"{model_prefix}_attn_window_{name}"
                self._force_attention_weights(module)
                module.register_forward_hook(
                    self._hook_factory(self.knowledge_attn, key, mode='output_index_1', detach=detach)
                )
            elif "inter_region_attn" in name:
                key = f"{model_prefix}_attn_region_{name}"
                self._force_attention_weights(module)
                module.register_forward_hook(
                    self._hook_factory(self.knowledge_attn, key, mode='output_index_1', detach=detach)
                )

            elif name.endswith('attn.attn_drop'):
                key = f"{model_prefix}_attn_{name.replace('.attn.attn_drop', '')}"
                module.register_forward_hook(
                    self._hook_factory(self.knowledge_attn, key, mode='input', detach=detach)
                )
            
            elif isinstance(module, nn.MultiheadAttention):
                key = f"{model_prefix}_attn_{name}"
                self._force_attention_weights(module)
                module.register_forward_hook(
                    self._hook_factory(self.knowledge_attn, key, mode='output_index_1', detach=detach)
                )

            # if model_prefix == 'labram' and name.endswith('pos_drop'):
            if model_prefix == 'labram' and name.endswith('patch_embed'):
                key = f"{model_prefix}_emb"
                module.register_forward_hook(
                    self._hook_factory(self.knowledge_emb, key, mode='output', detach=detach)
                )
            
            elif model_prefix != 'labram' and (name.endswith('patch_embedding') or name.endswith('patch_embed')):
                key = f"{model_prefix}_emb"
                module.register_forward_hook(
                    self._hook_factory(self.knowledge_emb, key, mode='output', detach=detach)
                )

    def _hook_factory(self, storage, key, mode, detach):
        def hook(module, inp, out):
            if mode == 'input':
                val = inp[0]
            elif mode == 'output_index_1':
                val = out[1]
            else: # mode == 'output'
                val = out
            storage[key] = val.detach() if detach else val
        return hook

    def _force_attention_weights(self, m):
        import inspect
        orig_forward = m.forward
        
        # 检查 forward 函数是否接受 average_attn_weights 参数
        sig = inspect.signature(orig_forward)

        def new_forward(*args, **kwargs):
            kwargs['need_weights'] = True
            if 'average_attn_weights' in sig.parameters:
                kwargs['average_attn_weights'] = False
            return orig_forward(*args, **kwargs)
            
        m.forward = new_forward

    def flush(self):
        self.knowledge_attn.clear()
        self.knowledge_emb.clear()
