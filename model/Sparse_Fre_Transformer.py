"""
DT
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')


import math
import torch
import torch.nn.functional as F
from torch import nn
from utils.model_utils import calculate_unfold_output_length, denormalize
from model.Blocks import Block,CLSHead,PatchEmbedding,\
    ForecastHead,LearnablePositionalEmbedding


class DynamicLinear(nn.Module):
    """
    A dynamic linear layer that can interpolate the weight size to support any given input and output feature dimension.
    """

    def __init__(self, in_features=None, out_features=None, fixed_in=0, bias=True):
        super(DynamicLinear, self).__init__()
        assert fixed_in < in_features, "fixed_in < in_features is required !!!"
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.fixed_in = fixed_in

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, out_features):
        """
        Forward pass for the dynamic linear layer.
        """
        fixed_weights = self.weights[:, :self.fixed_in]
        dynamic_weights = self.weights[:, self.fixed_in:]
        this_bias = self.bias
        in_features = x.shape[-1]

        if in_features != self.weights.size(1) or out_features != self.weights.size(0):
            dynamic_weights = F.interpolate(dynamic_weights.unsqueeze(0).unsqueeze(0), size=(
                out_features, in_features-self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            if self.fixed_in != 0:
                fixed_weights = F.interpolate(fixed_weights.unsqueeze(0).unsqueeze(0), size=(
                    out_features, self.fixed_in), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        if out_features != self.weights.size(0):
            this_bias = F.interpolate(this_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), size=(
                1, out_features), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).squeeze(0)
        return F.linear(x, torch.cat((fixed_weights, dynamic_weights), dim=1), this_bias)


class Model(nn.Module):
    def __init__(self, args, task_data_config_list):
        super(Model, self).__init__()
        self.args = args
        self.configs_list = task_data_config_list
        self.num_task = len(task_data_config_list)
        self._init_tokens()
        self._init_cls_num()
        self._init_num()
        self._init_blocks()
        self._init_task_submodel()
        # self._init_prompt_handler()
        
        
    def _init_tokens(self):
        self.prompt_tokens = nn.ParameterDict({})
        self.mask_tokens = nn.ParameterDict({})
        self.cls_tokens = nn.ParameterDict({})
        self.category_tokens = nn.ParameterDict({})
        
        for i in range(self.num_task):
            dataset_name = self.configs_list[i][1]['dataset_name']
            task_data_name = self.configs_list[i][0]
            if dataset_name not in self.prompt_tokens:
                self.prompt_tokens[dataset_name] = torch.randn(1, self.configs_list[i][1]['enc_in'], # prompt
                                                               self.args.prompt_num, self.args.d_model) * 0.02
                self.mask_tokens[dataset_name] = torch.zeros(
                    1, self.configs_list[i][1]['enc_in'], 1, self.args.d_model)

            if self.configs_list[i][1]['task_name'] == 'Classification':
                self.category_tokens[task_data_name] = torch.randn(
                    1, self.configs_list[i][1]['enc_in'], self.configs_list[i][1]['num_class'], self.args.d_model) * 0.02

                self.cls_tokens[task_data_name] = torch.randn(
                    1, self.configs_list[i][1]['enc_in'], 1, self.args.d_model) * 0.02
        
    def _init_cls_num(self):
        self.cls_nums = {}
        for i in range(self.num_task):
            task_data_name = self.configs_list[i][0]
            task_config = self.configs_list[i][1]
            
            if task_config['task_name'] == 'Classification':
                self.cls_nums[task_data_name] = task_config['num_class']
            elif task_config['task_name'] == 'Forecasting':
                self.cls_nums[task_data_name] = self._calculate_forecast_cls_num(task_config)

    def _calculate_forecast_cls_num(self, task_config): # TODO check
        seq_len = task_config['seq_len']
        pred_len = task_config['pred_len']
        patch_len = self.args.patch_len
        stride = self.args.stride

        padding = (patch_len - seq_len % patch_len) % patch_len
        input_token_windows_len = calculate_unfold_output_length(seq_len + padding, stride, patch_len)
        input_pad = stride * (input_token_windows_len - 1) + patch_len - seq_len
        pred_token_windows_len = calculate_unfold_output_length(pred_len - input_pad, stride, patch_len)
        real_len = seq_len + pred_len
        
        return [pred_token_windows_len, pred_len, real_len]
    
    def _init_num(self):
        self.prompt_num = self.args.prompt_num
        self.stride = self.args.stride
        self.pad = self.args.stride
        self.patch_len = self.args.patch_len        
            
    def _init_blocks(self):
        args = self.args
        self.patch_embeddings = PatchEmbedding(
            self.args.d_model,
            self.args.patch_len,
            self.args.stride,
            self.args.stride,
            self.args.dropout)
        self.position_embedding = LearnablePositionalEmbedding(self.args.d_model)
        self.prompt2forecat = DynamicLinear(128, 128, fixed_in=self.args.prompt_num)

        # basic blocks
        self.block_num = self.args.e_layers
        # self.blocks = nn.ModuleList(
        #     [Block(dim=self.args.d_model, num_heads=self.args.n_heads, qkv_bias=False, qk_norm=False,
        #                 mlp_ratio=8., proj_drop=self.args.dropout, attn_drop=0., drop_path=0.,
        #                 init_values=None, prefix_token_length=self.args.prompt_num) for l in range(self.args.e_layers)]
        # )       
        
        self.blocks = nn.ModuleList([
            Block(    mixing_type='afno', 
                  double_skip=True, 
                  width=32, 
                  n_blocks=4, 
                  mlp_ratio=1., 
                  channel_first=True, 
                  modes=32, 
                  act='gelu',
                  )
            for i in range(args.e_layers)])
        
        self.cls_head = CLSHead(args.d_model, head_dropout=args.dropout)
        self.forecast_head = ForecastHead(
            args.d_model, args.patch_len, args.stride, args.stride, prefix_token_length=args.prompt_num, head_dropout=args.dropout)    
        
    def tokenize(self, x, mask=None):
        # 计算均值并进行去均值操作
        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means

        # 计算标准差
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x * x, dim=1) /
                               torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(dim=1)
        else:
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        
        # stdev = stdev.unsqueeze(dim=1)
        x /= stdev

        # 重新排列维度
        x = x.permute(0, 2, 1)

        # 计算需要的填充量并进行填充
        remainder = x.shape[2] % self.patch_len
        padding = self.patch_len - remainder if remainder != 0 else 0
        if padding > 0:
            x = F.pad(x, (0, padding))

        # 进行嵌入操作
        x, n_vars = self.patch_embeddings(x)

        return x, means, stdev, n_vars, padding
    
    def _init_task_submodel(self):
        self.task_to_method = {
            # 'Forecasting': self.forecast,
            'Forecasting': self.forecast,
            'Classification': self.classification,
            'Imputation': self.imputation,
            'Anomaly_detection': self.anomaly_detection
        }
    
    def classification(self, x, task_id):
        # 获取数据集名称和任务数据名称
        dataset_name = self.configs_list[task_id][1]['dataset_name']
        task_data_name = self.configs_list[task_id][0]

        # 获取前缀提示和任务提示
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.cls_tokens[task_data_name]
        task_prompt_num = 1
        category_token = self.category_tokens[task_data_name]

        # 对输入数据进行标记化处理
        x, means, stdev, n_vars, _ = self.tokenize(x) # B,L,C 

        # 获取序列长度
        seq_len = x.shape[-2]

        # 准备提示信息
        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name='Classification'
        )

        # 通过主干网络处理
        x = self.backbone(x, prefix_prompt.shape[2], seq_len)

        # 通过分类头处理
        x = self.cls_head(x, category_token)

        return x
    
    def imputation(self, x, mask, task_id):
        # 获取数据集名称
        dataset_name = self.configs_list[task_id][1]['dataset']

        # 获取前缀提示和任务提示
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.mask_tokens[dataset_name]

        # 获取序列长度
        seq_len = x.shape[1]

        # 对输入数据进行标记化处理
        x, means, stdev, n_vars, padding = self.tokenize(x, mask)

        # 准备提示信息
        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, None, mask=mask, task_name='imputation'
        )

        # 获取序列标记长度
        seq_token_len = x.shape[-2] - prefix_prompt.shape[2]

        # 通过主干网络处理
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        # 通过预测头处理
        x = self.forecast_head(x, seq_len + padding, seq_token_len)

        # 截取到原始序列长度
        x = x[:, :seq_len]
        
        x = denormalize(x, means, stdev)

        return x

    def forecast(self, x, task_id):
        # 获取数据集名称和任务数据名称
        dataset_name = self.configs_list[task_id][1]['dataset_name']
        task_data_name = self.configs_list[task_id][0]

        # 获取前缀提示和任务提示
        prefix_prompt = self.prompt_tokens[dataset_name]
        task_prompt = self.mask_tokens[dataset_name]
        task_prompt_num = self.cls_nums[task_data_name][0]
        task_seq_num = self.cls_nums[task_data_name][1]
        real_seq_len = self.cls_nums[task_data_name][2]

        # 对输入数据进行标记化处理
        x, means, stdev, n_vars, _ = self.tokenize(x)

        # 准备提示信息
        x = self.prepare_prompt(
            x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name='forecast'
        )

        # 获取序列标记长度
        seq_token_len = x.shape[-2] - prefix_prompt.shape[2]

        # 通过主干网络处理
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        # 通过预测头处理
        x = self.forecast_head(x, real_seq_len, seq_token_len)

        # 截取到任务序列长度
        x = x[:, -task_seq_num:]

        # 反归一化处理
        x = denormalize(x, means, stdev)

        return x

    def anomaly_detection(self, x, task_id):
        # 获取数据集名称
        dataset_name = self.configs_list[task_id][1]['dataset']
        prefix_prompt = self.prompt_tokens[dataset_name]

        # 获取序列长度
        seq_len = x.shape[1]

        # 对输入数据进行标记化处理
        x, means, stdev, n_vars, padding = self.tokenize(x)

        # 准备提示信息
        x = self.prepare_prompt(x, n_vars, prefix_prompt, None, None, task_name='anomaly_detection')

        # 获取序列标记长度
        seq_token_len = x.shape[-2] - prefix_prompt.shape[2]

        # 通过主干网络处理
        x = self.backbone(x, prefix_prompt.shape[2], seq_token_len)

        # 通过预测头处理
        x = self.forecast_head(x, seq_len + padding, seq_token_len)

        # 截取到原始序列长度
        x = x[:, :seq_len]

        # 反归一化处理
        x = denormalize(x, means, stdev)

        return x
    
    # def _init_prompt_handler(self):
    #     def add_position_embedding(x):
    #         return x + self.position_embedding(x)

    #     def add_prompt_tokens(x, prompt, dim=-2):
    #         return torch.cat((prompt, x), dim=dim)

    #     def handle_forecast(x, this_prompt, task_prompt, task_prompt_num):
    #         this_mask_prompt = task_prompt.repeat(x.shape[0], 1, task_prompt_num, 1)
    #         init_full_input = add_prompt_tokens(x, this_prompt)
    #         init_full_input = add_prompt_tokens(init_full_input, this_mask_prompt)
    #         init_mask_prompt = self.prompt2forecat(init_full_input.transpose(-1, -2), init_full_input.shape[2] - this_prompt.shape[2]).transpose(-1, -2)
    #         this_function_prompt = init_mask_prompt[:, :, -task_prompt_num:]
    #         x = add_prompt_tokens(x, this_prompt)
    #         x = add_prompt_tokens(x, this_function_prompt)
    #         x[:, :, self.prompt_num:] = add_position_embedding(x[:, :, self.prompt_num:])
    #         return x

    #     def handle_Classification(x, this_prompt, task_prompt):
    #         this_function_prompt = task_prompt.repeat(x.shape[0], 1, 1, 1)
    #         x = add_position_embedding(x)
    #         x = add_prompt_tokens(x, this_prompt)
    #         x = add_prompt_tokens(x, this_function_prompt)
    #         return x

    #     def handle_imputation(x, this_prompt, task_prompt, mask):
    #         mask = 1 - mask
    #         mask = mask.permute(0, 2, 1)
    #         mask = self.mark2token(mask)
    #         mask_repeat = mask.unsqueeze(dim=-1).repeat(1, 1, 1, x.shape[-1])
    #         mask_token = task_prompt
    #         x = x * (1 - mask_repeat) + mask_token * mask_repeat
    #         init_full_input = add_prompt_tokens(x, this_prompt)
    #         init_mask_prompt = self.prompt2forecat(init_full_input.transpose(-1, -2), x.shape[2]).transpose(-1, -2)
    #         x = x * (1 - mask_repeat) + init_mask_prompt * mask_repeat
    #         x = add_position_embedding(x)
    #         x = add_prompt_tokens(x, this_prompt)
    #         return x

    #     def handle_anomaly_detection(x, this_prompt):
    #         x = add_position_embedding(x)
    #         x = add_prompt_tokens(x, this_prompt)
    #         return x
            
    #     self.task_handlers = {
    #     'forecast': handle_forecast,
    #     'Classification': handle_Classification,
    #     'imputation': handle_imputation,
    #     'anomaly_detection': handle_anomaly_detection
    # }

    # def handle_task(self, task_type, **kwargs):
    #     if task_type in self.task_handlers:
    #         return self.task_handlers[task_type](**kwargs)
    #     else:
    #         raise ValueError(f"Unknown task type: {task_type}")
        
    def prepare_prompt(self, x, n_vars, prefix_prompt, task_prompt, task_prompt_num, task_name=None, mask=None):
        x = torch.reshape(
            x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        # append prompt tokens
        this_prompt = prefix_prompt.repeat(x.shape[0], 1, 1, 1)

        if task_name == 'forecast':
            this_mask_prompt = task_prompt.repeat(
                x.shape[0], 1, task_prompt_num, 1)
            init_full_input = torch.cat(
                (this_prompt, x, this_mask_prompt), dim=-2)
            init_mask_prompt = self.prompt2forecat(init_full_input.transpose(
                -1, -2), init_full_input.shape[2]-prefix_prompt.shape[2]).transpose(-1, -2)
            this_function_prompt = init_mask_prompt[:, :, -task_prompt_num:]
            x = torch.cat((this_prompt, x, this_function_prompt), dim=2)
            x[:, :, self.prompt_num:] = x[:, :, self.prompt_num:] + \
                self.position_embedding(x[:, :, self.prompt_num:])
        elif task_name == 'Classification':
            this_function_prompt = task_prompt.repeat(x.shape[0], 1, 1, 1)
            x = x + self.position_embedding(x)
            x = torch.cat((this_prompt, x, this_function_prompt), dim=2) # B,C,PROMPT.D  concate x  concate B,C,1,D 
        elif task_name == 'imputation':
            # fill the masked parts with mask tokens
            # for imputation, masked is 0, unmasked is 1, so here to reverse mask
            mask = 1-mask
            mask = mask.permute(0, 2, 1)
            mask = self.mark2token(mask)
            mask_repeat = mask.unsqueeze(dim=-1)

            mask_token = task_prompt
            mask_repeat = mask_repeat.repeat(1, 1, 1, x.shape[-1])
            x = x * (1-mask_repeat) + mask_token * mask_repeat

            init_full_input = torch.cat((this_prompt, x), dim=-2)
            init_mask_prompt = self.prompt2forecat(
                init_full_input.transpose(-1, -2), x.shape[2]).transpose(-1, -2)
            # keep the unmasked tokens and fill the masked ones with init_mask_prompt.
            x = x * (1-mask_repeat) + init_mask_prompt * mask_repeat
            x = x + self.position_embedding(x)
            x = torch.cat((this_prompt, x), dim=2)
        elif task_name == 'anomaly_detection':
            x = x + self.position_embedding(x)
            x = torch.cat((this_prompt, x), dim=2)

        return x   
    
    def backbone(self, x, prefix_len, seq_len):
        attn_mask = None
        for block in self.blocks:
            x = block(x, prefix_seq_len=prefix_len +
                      seq_len, attn_mask=attn_mask)
        return x
    
    def random_masking(self, x, min_mask_ratio, max_mask_ratio):
        """
        Perform per-sample random masking.
        """
        N, V, L, D = x.shape  # batch, var, length, dim

        # Calculate mask ratios and lengths to keep for each sample in the batch
        mask_ratios = torch.rand(N, device=x.device) * \
            (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)

        # Create a range tensor and compare with len_keeps for mask generation
        range_tensor = torch.arange(L, device=x.device).expand(N, L)
        mask = (range_tensor >= len_keeps.unsqueeze(1))

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask = mask.float()

        return mask

    def right_masking(self, x, min_mask_ratio, max_mask_ratio):
        N, V, L, D = x.shape  # batch, var, length, dim

        # Randomly choose a mask ratio for each sample within the specified range
        mask_ratios = torch.rand(N, device=x.device) * \
            (max_mask_ratio - min_mask_ratio) + min_mask_ratio
        len_keeps = (L * (1 - mask_ratios)).long()

        # Binary mask creation without a for loop
        len_keeps_matrix = len_keeps.unsqueeze(1).expand(N, L)
        indices = torch.arange(L, device=x.device).expand_as(len_keeps_matrix)
        mask = indices >= len_keeps_matrix
        mask = mask.float()

        return mask

    def choose_masking(self, x, right_prob, min_mask_ratio, max_mask_ratio):
        # Generate a random number to decide which masking function to use
        if torch.rand(1).item() > right_prob:
            return self.random_masking(x, min_mask_ratio, max_mask_ratio)
        else:
            return self.right_masking(x, min_mask_ratio, max_mask_ratio)

    def get_mask_seq(self, mask, seq_len):
        mask_seq = mask.unsqueeze(dim=-1).repeat(1, 1, self.patch_len)
        mask_seq = mask_seq.permute(0, 2, 1)
        mask_seq = mask_seq.masked_fill(mask_seq == 0, -1e9)
        # Fold operation
        mask_seq = torch.nn.functional.fold(mask_seq, output_size=(
            seq_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1))
        # Apply threshold to bring back to 0/1 values
        mask_seq = (mask_seq > 0).float()
        mask_seq = mask_seq.squeeze(dim=-1).squeeze(dim=1)
        return mask_seq    

    def cat_grid_1d(self, x): # TODO + forward
        B, L, C = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(torch.linspace(0, 1, L), dtype=torch.float)
        gridx = gridx.reshape(1, L, 1).repeat([B, 1, C])
        x = torch.cat((x, gridx), dim=-1).contiguous()
        return x   
    
    def forward(self, x_enc, mask=None, task_id=None, task_name=None):
        if task_name in self.task_to_method:
            method = self.task_to_method[task_name]
            if task_name == 'imputation':
                dec_out = method(x_enc, mask, task_id)
            else:
                dec_out = method(x_enc, task_id)
            return dec_out  # [B, L, D] or CLS is [B, N]
        else:
            raise ValueError(f"Unknown task name: {task_name}")
    

#%% test
if __name__ == '__main__':
    import sys
    import os
    # 将当前工作目录添加到 Python 模块搜索路径中
    sys.path.append('/home/user/LQ/B_Signal/Signal_foundation_model/Foundation_Model_Digital_twin/')
    import torch
    import yaml
    from utils.dataset_utils import read_task_data_config,get_task_data_config_list
    from types import SimpleNamespace
    # 假设配置文件名为 config.yaml
    config_file = 'model/config.yaml'

    # 读取配置文件
    with open(config_file, 'r') as file:
        args = yaml.safe_load(file)
        
    args = SimpleNamespace(**args)
    
    configs_list = read_task_data_config(args.task_data_config_path)
    configs_list = get_task_data_config_list(
            configs_list, default_batch_size=args.batch_size)

    # 初始化 UniTS 模型
    model = Model(args, configs_list)

    # 创建输入数据
    B, L, C = 2, 2048, 2
    x_enc = torch.randn(B, L, C)
    x_mark_enc = torch.randn(B, L, C)
    # 假设任务名称为 'Forecasting'
    task_name = 'Forecasting' # Forecasting classification
    task_id = 0  # 假设任务ID为0

    output = model.forward(x_enc, x_mark_enc, task_id=task_id, task_name=task_name)
    print(output.shape)  # torch.Size([2, 1]