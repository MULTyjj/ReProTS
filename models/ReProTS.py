from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            from transformers import LlamaConfig, LlamaModel, LlamaTokenizer
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("Local LLAMA model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local LLAMA tokenizer files not found. Attempting to download them...")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif configs.llm_model == 'GPT2':
            from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local GPT2 model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local GPT2 tokenizer files not found. Attempting to download them...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif configs.llm_model == 'BERT':
            from transformers import BertConfig, BertModel, BertTokenizer
            bert_model_path = "/root/ReProTS-main/bert"
            self.bert_config = BertConfig.from_pretrained(bert_model_path)
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    bert_model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError as e:
                raise FileNotFoundError(
                    f"Local BERT model files not found in {bert_model_path}. Please ensure the correct path is provided and all required files exist.\nError: {str(e)}"
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    bert_model_path,
                    trust_remote_code=True,
                    local_files_only=False
                )
            except EnvironmentError as e:
                raise FileNotFoundError(
                    f"Local BERT tokenizer files not found in {bert_model_path}. Please ensure the correct path is provided and all required files exist.\nError: {str(e)}"
                )

        elif configs.llm_model == 'DEEPSEEK':
            from transformers import AutoTokenizer, AutoModelForCausalLM
            deepseek_model_path = "/root/ReProTS-main/deepseek"
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    deepseek_model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            except EnvironmentError:
                print("本地 DeepSeek 模型文件未找到，尝试从 HuggingFace 下载...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    "deepseek-ai/deepseek-coder-6.7b-base",
                    trust_remote_code=True,
                    local_files_only=False,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    deepseek_model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except EnvironmentError:
                print("本地 DeepSeek tokenizer 未找到，尝试从 HuggingFace 下载...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/deepseek-coder-6.7b-base",
                    trust_remote_code=True,
                    local_files_only=False,
                )

        else:
            raise ValueError(f"Unsupported LLM model: {configs.llm_model}")


        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # 将所有输入张量移动到 GPU（如果可用）
        x_enc = x_enc.to(device)
        x_mark_enc = x_mark_enc.to(device)
        x_dec = x_dec.to(device)
        x_mark_dec = x_mark_dec.to(device)

        # 数据标准化
        x_enc = self.normalize_layers(x_enc, 'norm')

        # 获取批量大小、时间步长、特征数量
        B, T, N = x_enc.size()

        # 将数据重塑为适合的形状
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        # 计算常见的统计量
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        # 计算额外的统计量
        std_dev = torch.std(x_enc, dim=1)  # 标准差
        autocorr = self.calculate_autocorrelation(x_enc, lag=1)  # 自相关系数（滞后1期）
        skewness = self.calculate_skewness(x_enc)  # 偏度
        kurtosis = self.calculate_kurtosis(x_enc)  # 峰度
        delta_min_max = max_values - min_values  # 最大值和最小值的差异
        rate_of_change = self.calculate_rate_of_change(x_enc, window_size=5)  # 最近5期的变化率

        # 构建提示词
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            std_dev_str = str(std_dev[b].tolist()[0])
            autocorr_str = str(autocorr[b])
            skewness_str = str(skewness[b].tolist()[0])
            kurtosis_str = str(kurtosis[b].tolist()[0])
            delta_min_max_str = str(delta_min_max[b].tolist()[0])
            rate_of_change_str = str(rate_of_change[b])

            # 构建提示词
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description} "
                f"Task description: forecast the next {str(self.pred_len)} steps based on the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"standard deviation {std_dev_str}, "
                f"trend of the input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"autocorrelation at lag 1 {autocorr_str}, "
                f"skewness {skewness_str}, "
                f"kurtosis {kurtosis_str}, "
                f"change in min-max range {delta_min_max_str}, "
                f"rate of change (last 5 steps) {rate_of_change_str}, "
                f"top 5 lags are: {lags_values_str} "
                "<|end_prompt|>"
            )

            prompt.append(prompt_)

        # 恢复 x_enc 的形状
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # 使用 tokenizer 编码提示词
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        # 获取源嵌入层并计算嵌入
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        # 编码时间序列数据
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        
        # 拼接提示词的嵌入与时间序列的嵌入
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        # 重塑输出
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # 进行输出投影
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # 恢复输出的标准化
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out
    def calculate_skewness(self, x_enc):
        """
        手动计算偏度，直接在GPU上计算
        """
        # 计算均值和标准差
        mean = torch.mean(x_enc, dim=1, keepdim=True)
        std = torch.std(x_enc, dim=1, keepdim=True)

        # 计算偏度
        skewness = torch.mean(((x_enc - mean) / std) ** 3, dim=1)

        return skewness

    # 辅助函数：计算自相关系数
    def calculate_autocorrelation(self, x_enc, lag=1):
        """
        计算给定时间序列的自相关系数
        """
        autocorr = torch.corrcoef(x_enc.view(-1, x_enc.size(1)).t())[lag]
        return autocorr

    # 辅助函数：计算峰度
    def calculate_kurtosis(self, x_enc):
        """
        手动计算峰度（Kurtosis）
        """
        # 计算均值和标准差
        mean = torch.mean(x_enc, dim=1, keepdim=True)  # 均值
        std = torch.std(x_enc, dim=1, keepdim=True)    # 标准差

        # 利用均值和标准差进行归一化，然后计算峰度
        normalized = (x_enc - mean) / std  # 归一化，避免重复计算
        kurtosis = torch.mean(normalized ** 4, dim=1) - 3  # 计算峰度

        return kurtosis


    # 辅助函数：计算变化速率
    def calculate_rate_of_change(self, x_enc, window_size=5):
        """
        计算给定时间序列在过去 N 个时间步内的变化速率
        """
        # 计算时间序列的变化
        changes = torch.diff(x_enc, dim=1)  # 计算相邻时间步的差异，形状为 (B, T-1, N)

        # 取每个时间序列的最后 `window_size` 个变化值
        changes = changes[:, -window_size:, :]  # 只保留最后 `window_size` 步的变化

        # 计算变化率（每个批次的平均变化率）
        rate_of_change = torch.mean(changes, dim=1)  # 计算每个样本的变化速率，结果形状为 (B, N)

        return rate_of_change

    # 辅助函数：计算滞后值

    def calcute_lags(self, x_enc, device='cuda'):
        window_size = 10
        step = 1

        B, T, N = x_enc.shape
        lags = []

        # Move data to GPU if available
        x_enc = x_enc.to(device)
        x_enc = x_enc.permute(0, 2, 1)  # Shape: (B, N, T)

        kernel_size = window_size
        stride = step

        conv = torch.nn.Conv1d(in_channels=N, out_channels=N, kernel_size=kernel_size, stride=stride, padding=0, bias=False).to(device)

        for b in range(B):
            series = x_enc[b]  # Shape: (N, T)

            # Use sliding windows without redundant calculation
            attention_scores = conv(series.unsqueeze(0)).squeeze(0)  # Shape: (N, T')

            attention_scores = attention_scores.mean(dim=0)  # Shape: (T')

            # Find top-k attention scores
            top_scores, top_indices = torch.topk(attention_scores, self.top_k, dim=0)

            lags.append(top_indices)

        lags = torch.stack(lags, dim=0)  # Shape: (B, top_k)
        return lags





class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        # 原始的线性变换
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

        # 多尺度映射
        self.query_projection_small = nn.Linear(d_model, d_keys * n_heads)  # 小尺度
        self.key_projection_small = nn.Linear(d_llm, d_keys * n_heads)  # 小尺度
        self.value_projection_small = nn.Linear(d_llm, d_keys * n_heads)  # 小尺度

        self.query_projection_large = nn.Linear(d_model, d_keys * n_heads)  # 大尺度
        self.key_projection_large = nn.Linear(d_llm, d_keys * n_heads)  # 大尺度
        self.value_projection_large = nn.Linear(d_llm, d_keys * n_heads)  # 大尺度

        # 加权平均的可学习权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 小尺度的权重
        self.beta = nn.Parameter(torch.tensor(0.5))   # 大尺度的权重

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # 计算多尺度嵌入
        target_embedding_small = self.query_projection_small(target_embedding).view(B, L, H, -1)
        source_embedding_small = self.key_projection_small(source_embedding).view(S, H, -1)
        value_embedding_small = self.value_projection_small(value_embedding).view(S, H, -1)

        target_embedding_large = self.query_projection_large(target_embedding).view(B, L, H, -1)
        source_embedding_large = self.key_projection_large(source_embedding).view(S, H, -1)
        value_embedding_large = self.value_projection_large(value_embedding).view(S, H, -1)

        # 计算多尺度的注意力
        out_small = self.reprogramming(target_embedding_small, source_embedding_small, value_embedding_small)
        out_large = self.reprogramming(target_embedding_large, source_embedding_large, value_embedding_large)

        # 确保输出维度一致，调整小尺度结果的维度与大尺度一致
        if out_small.shape[3] != out_large.shape[3]:
            out_small = out_small.view(B, L, H, -1)  # 调整为与大尺度相同的形状

        # 加权平均合并
        out = self.alpha * out_small + self.beta * out_large  # 使用加权平均

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / torch.sqrt(torch.tensor(E, dtype=torch.float32))

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

