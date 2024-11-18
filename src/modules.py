import torch
from torch import nn
import torch.nn.functional as F


class SvegaLinear(nn.Module):
    def __init__(self, input_dimension, output_dimension, bias=True):
        super().__init__()

        self.weights = nn.Parameter(torch.empty(input_dimension, output_dimension))
        nn.init.kaiming_uniform_(self.weights)

        self.bias = bias

        if self.bias:
            self.biases = nn.Parameter(torch.empty(output_dimension))
            nn.init.zeros_(self.biases)
        
    def forward(self, X):
        if self.bias:
            return X @ self.weights + self.biases
        return X @ self.weights
    

class SvegaSlowConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_height = kernel_size
        self.kernel_width = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_height, self.kernel_width))
        nn.init.kaiming_uniform_(self.kernel)

    def forward(self, X):
        B, _, H, W = X.shape

        X_padded = F.pad(X, (self.padding, self.padding, self.padding, self.padding), "constant", 0)

        feature_map_height = int((H + 2 * self.padding - self.kernel_height) / self.stride) + 1
        feature_map_width = int((W + 2 * self.padding - self.kernel_width) / self.stride) + 1

        feature_map = torch.zeros((B, self.out_channels, feature_map_height, feature_map_width))

        for batch in range(B):
            for out_channel in range(self.out_channels):
                for row in range(feature_map_height):
                    for col in range(feature_map_width):
                        input_tensor = X_padded[batch, 0:self.in_channels, row * self.stride:row * self.stride + self.kernel_height, col * self.stride:col * self.stride + self.kernel_width]
                        kernel_tensor = self.kernel[out_channel]
                        feature_map[batch, out_channel, row, col] = torch.sum(input_tensor * kernel_tensor)

        return feature_map
    

class SvegaFastConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_height = kernel_size
        self.kernel_width = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = nn.Parameter(torch.empty(out_channels, in_channels * self.kernel_height * self.kernel_width))
        nn.init.kaiming_uniform_(self.kernel)

    def forward(self, X):
        B, _, H, W = X.shape

        X_padded = F.pad(X, (self.padding, self.padding, self.padding, self.padding), "constant", 0)

        patches = F.unfold(X_padded, kernel_size=(self.kernel_height, self.kernel_width), stride=self.stride)

        unfolded_feature_map = self.kernel @ patches

        feature_map_height = int((H + 2 * self.padding - self.kernel_height) / self.stride) + 1
        feature_map_width = int((W + 2 * self.padding - self.kernel_width) / self.stride) + 1

        feature_map = unfolded_feature_map.view(B, self.out_channels, feature_map_height, feature_map_width)

        return feature_map
    

class SvegaSlowMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()

        self.kernel_height = kernel_size
        self.kernel_width = kernel_size
        self.stride = stride

    def forward(self, X):
        B, C, H, W = X.shape

        output_map_height = int((H - self.kernel_height) / self.stride) + 1
        output_map_width = int((W - self.kernel_width) / self.stride) + 1

        output_map = torch.zeros((B, C, output_map_height, output_map_width))

        for batch in range(B):
            for channel in range(C):
                for row in range(output_map_height):
                    for col in range(output_map_width):
                        row_start = row * self.stride
                        row_end = row_start + self.kernel_height
                        col_start = col * self.stride
                        col_end = col_start + self.kernel_width
                        output_map[batch, channel, row, col] = torch.max(X[batch, channel, row_start:row_end, col_start:col_end])

        return output_map


class SvegaLeakyReLU(nn.Module):
    def __init__(self, alpha=0):
        super().__init__()

        self.alpha = alpha

    def forward(self, X):
        return torch.where(X > 0, X, X * self.alpha)
    

class SvegaFlatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.flatten(X, start_dim=1)
    

class SvegaLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-05):
        super().__init__()

        self.eps = eps

        self.scale_parameters = nn.Parameter(torch.ones(hidden_size))
        self.shift_parameters = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, X):
        means = torch.mean(X, dim=-1, keepdim=True)
        variances = torch.sqrt(X.var(dim=-1, keepdim=True, unbiased=False) + self.eps)

        normalized_X = (X - means) / variances
        scaled_X = normalized_X * self.scale_parameters
        shifted_x = scaled_X + self.shift_parameters
        
        return shifted_x
    

class SvegaBatchNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-05, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        self.scale_parameters = nn.Parameter(torch.ones(hidden_size))
        self.shift_parameters = nn.Parameter(torch.zeros(hidden_size))

        self.register_buffer("running_means", torch.zeros(hidden_size))
        self.register_buffer("running_variances", torch.zeros(hidden_size))

    def forward(self, X):
        if self.training:
            batch_means = torch.mean(X, dim=0, keepdim=True)
            batch_variances = torch.sqrt(X.var(dim=0, keepdim=True, unbiased=False) + self.eps)

            self.running_means = self.momentum * batch_means + (1 - self.momentum) * self.running_means
            self.running_variances = self.momentum * batch_variances + (1 - self.momentum) * self.running_variances

            normalized_X = (X - batch_means) / batch_variances
        else:
            normalized_X = (X - self.running_means) / torch.sqrt(self.running_variances + self.eps)

        scaled_X = normalized_X * self.scale_parameters
        shifted_x = scaled_X + self.shift_parameters
            
        return shifted_x


class SvegaDropout(nn.Module):
    def __init__(self, probability=0.2):
        super().__init__()

        self.probability = probability

    def forward(self, X):
        if self.training:
            probability_map = torch.full(X.shape, 1 - self.probability) 
            mask = torch.bernoulli(probability_map)
            zeroed_out_X = X * mask
            scaled_X = zeroed_out_X / (1 - self.probability)
            return scaled_X
        else:
            return X
        

class SvegaSelfAttentionHead(nn.Module):
    def __init__(self, head_size, sequence_length, embedding_size):
        super().__init__()

        self.head_size = head_size
        self.sequence_length = sequence_length

        self.key_layer = SvegaLinear(embedding_size, head_size, bias=False)
        self.query_layer = SvegaLinear(embedding_size, head_size, bias=False)
        self.value_layer = SvegaLinear(embedding_size, head_size, bias=False)

        self.dropout_layer = SvegaDropout(0.2)

        self.register_buffer("mask", torch.tril(torch.ones(sequence_length, sequence_length)))

    def forward(self, X):
        sequence_length = X.size(1)

        keys = self.key_layer(X)
        queries = self.query_layer(X)
        values = self.value_layer(X)
        
        attention_weights = queries @ keys.transpose(-2, -1) * (self.head_size ** -0.5)
        masked_attention_weights = attention_weights.masked_fill(self.mask[:sequence_length, :sequence_length] == 0, float("-inf"))
        attention_probabilities = F.softmax(masked_attention_weights, dim=-1)
        dropped_out_attention_probabilities = self.dropout_layer(attention_probabilities)
        attention_scores = dropped_out_attention_probabilities @ values
        
        return attention_scores


class SvegaMultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, head_size, sequence_length, embedding_size):
        super().__init__()

        self.self_attention_heads = nn.ModuleList([
            SvegaSelfAttentionHead(head_size, sequence_length, embedding_size) for _ in range(num_heads)
        ])

        self.projection_layer = nn.Linear(embedding_size, embedding_size)

    def forward(self, X):
        attention_output = torch.cat([self_attention_head(X) for self_attention_head in self.self_attention_heads], dim=-1)
        mixed_attention_output = self.projection_layer(attention_output)
        return mixed_attention_output
    

class SvegaFeedForwardLayer(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.stack = nn.Sequential(
            SvegaLinear(embedding_size, 4 * embedding_size),
            SvegaLeakyReLU(0.02),
            SvegaLinear(4 * embedding_size, embedding_size),
            SvegaDropout(0.2)
        )

    def forward(self, X):
        return self.stack(X)


class SvegaTransformerBlock(nn.Module):
    def __init__(self, num_heads, sequence_length, embedding_size):
        super().__init__()

        head_size = int(embedding_size / num_heads)

        self.multi_head_self_attention = SvegaMultiHeadSelfAttention(num_heads, head_size, sequence_length, embedding_size)
        self.feed_forward_layer = SvegaFeedForwardLayer(embedding_size)

        self.layer_norm_1 = SvegaLayerNorm(embedding_size)
        self.layer_norm_2 = SvegaLayerNorm(embedding_size)

    def forward(self, X):
        multi_head_attention_output = X + self.multi_head_self_attention(X)
        normalized_multi_head_attention_output = self.layer_norm_1(multi_head_attention_output)

        feed_forward_output = normalized_multi_head_attention_output + self.feed_forward_layer(normalized_multi_head_attention_output)
        normalized_feed_forward_output = self.layer_norm_2(feed_forward_output)

        return normalized_feed_forward_output
    

class SvegaGPT(nn.Module):
    def __init__(self, vocabulary_size, sequence_length, embedding_size, num_transformer_blocks, num_heads):
        super().__init__()

        self.sequence_length = sequence_length

        self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding_table = nn.Embedding(sequence_length, embedding_size)

        self.transformer_blocks = nn.Sequential(
            *[SvegaTransformerBlock(num_heads, sequence_length, embedding_size) for _ in range(num_transformer_blocks)]
        )

        self.language_model_head = nn.Linear(embedding_size, vocabulary_size)

    def forward(self, X):
        sequence_length = X.size(1)

        token_embedding = self.token_embedding_table(X)
        position_embedding = self.position_embedding_table(torch.arange(sequence_length))
        input_embedding = token_embedding + position_embedding

        transformer_output = self.transformer_blocks(input_embedding)
        
        logits = self.language_model_head(transformer_output)
    
        return logits

    def generate(self, x_batch, max_new_tokens):
        for _ in range(max_new_tokens):
            truncated_x_batch = x_batch[:, -self.sequence_length:]
            
            logits = self(truncated_x_batch)
            last_token_logits = logits[:, -1, :]

            probabilities = F.softmax(last_token_logits, dim=-1)

            next_tokens = torch.multinomial(probabilities, num_samples=1)
            x_batch = torch.cat((x_batch, next_tokens), dim=1)

        return x_batch
