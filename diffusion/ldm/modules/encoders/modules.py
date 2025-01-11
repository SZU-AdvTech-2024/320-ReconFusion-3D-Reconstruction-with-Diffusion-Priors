import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

from diffusion.ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=74,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


# class FrozenCLIPTextEmbedder(nn.Module):
#     """
#     Uses the CLIP transformer encoder for text.
#     """
#     def __init__(self, version='ViT-L/14', device="cuda", max_length=74, n_repeat=1, normalize=True):
#         super().__init__()
#         self.model, _ = clip.load(version, jit=False, device="cpu")
#         self.device = device
#         self.max_length = max_length
#         self.n_repeat = n_repeat
#         self.normalize = normalize
#
#     def freeze(self):
#         self.model = self.model.eval()
#         for param in self.parameters():
#             param.requires_grad = False
#
#     def forward(self, text):
#         tokens = clip.tokenize(text).to(self.device)
#         z = self.model.encode_text(tokens)
#         if self.normalize:
#             z = z / torch.linalg.norm(z, dim=1, keepdim=True)
#         return z
#
#     def encode(self, text):
#         z = self(text)
#         if z.ndim==2:
#             z = z[:, None, :]
#         z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
#         return z

class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text and returns embeddings with shape [batch_size, 74, 768].
    """

    def __init__(self, version='ViT-L/14', device="cuda", max_length=74, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device=device)  # 加载 CLIP 模型
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        """冻结模型，使得所有参数不可训练"""
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        # Tokenize the text with a fixed max_length
        tokens = clip.tokenize(text, truncate=True).to(self.device)  # Tokenize并将数据转移到正确的设备上

        # 获取token的嵌入：确保返回形状为 [batch_size, seq_len, embedding_dim]
        with torch.no_grad():
            x = self.model.token_embedding(tokens)  # Shape: [batch_size, seq_len, embedding_dim]

            # Apply positional embedding
            # 这个位置编码操作确保位置编码会应用到每个token的嵌入上
            x = x + self.model.positional_embedding[:x.size(1), :].to(
                self.device)  # Shape: [batch_size, seq_len, embedding_dim]
            target_dtype = next(self.model.transformer.parameters()).dtype
            x = x.to(dtype=target_dtype)
            # Transformer processing
            x = x.permute(1, 0, 2)  # Switch to [seq_len, batch_size, embedding_dim]
            x = self.model.transformer(x)  # Output shape: [seq_len, batch_size, embedding_dim]
            x = x.permute(1, 0, 2)  # Back to [batch_size, seq_len, embedding_dim]

        # 截断/扩展至最大长度
        x = x[:, :self.max_length, :]  # Ensure the output length is max_length (74)

        # Normalize embeddings if needed
        if self.normalize:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:  # 确保 z 具有正确的维度
            z = z[:, None, :]  # 如果是 2D 形状则添加一个维度
        z = z.repeat(1, self.n_repeat, 1)  # 重复嵌入，生成 n_repeat 个重复的嵌入
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        # # 定义线性层，将768维映射到1280维
        # self.linear_layer = nn.Linear(768, 1280)

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        x = self.preprocess(x)
        embeddings = self.model.encode_image(x).to(torch.float32)  # 输出维度 [batch_size, 768]
        # adjusted_embeddings = self.linear_layer(embeddings)  # 调整到 [batch_size, 1280]
        return embeddings


if __name__ == "__main__":
    from ldm.util import count_params
    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)