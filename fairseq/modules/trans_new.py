# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import wandb

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        '''----------------------------Embeded&Position-----------------------------------'''
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)  
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8) 

        '''----------------------------Self Attn.&Add_Norm-----------------------------------'''
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        ) # 공통 드롭아웃. 어텐션 출력이나 FFN 출력에 사용.

        '''----------------------------FFN&Add_Norm-----------------------------------'''
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout (FFN)
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1( # 1st FFN
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2( # 2nd FFN
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim) #FFN용 layer_norm

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
    
    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        """--------------------------------Self Attn. & AddNorm--------------------------------------"""
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        """--------------------------------FFN. & AddNorm--------------------------------------"""
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()

        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False) #false이면, self?

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix

        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
            #추가
            self.encoder_attn_vis = None
            self.encoder_attn_vis_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
            #추가
            self.encoder_attn_vis = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_vis_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False
        
        #추가
        self.fuse_proj = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.fuse_dropout = nn.Dropout(getattr(args, "dropout", 0.1))

        self.lambda_param = nn.Parameter(torch.tensor(0.5))

        #entropy
        self.alpha = nn.Parameter(torch.tensor(3.0), requires_grad=False)  # 민감도(튜닝 후 필요하면 학습 가능 True)
        self._entropy_eps = 1e-8  # log 안정화용
        self.vis_dim_adapter = None

        '''
        # Shared latent alignment projection
        self.align_proj = SharedLatentAlign(
            txt_dim=args.encoder_embed_dim,
            vis_dim=args.encoder_embed_dim,  # 둘 다 hidden_dim 맞춰짐
            hidden_dim=self.embed_dim,
            proj_dim=self.embed_dim // 2
        )
        self.align_alpha = getattr(args, "align_alpha", 0.1)
        '''


    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
    
    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )
    
    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    '''------------------------------- Decoder ------------------------------------------'''
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        #추가
        vision_out: Optional[torch.Tensor] = None,
        vision_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        '''------------------------------- Self Attn. -----------------------------------'''
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # 이전 값이 존재할 때,
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        # cross-self-attention 모드이면서, 지금이 첫 스텝일 때만 실행
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)


        '''---------------------------- Cross Attn. ------------------------------'''
        # (Cross-Attn 블록 시작)
        # self_attn_layer_norm 는 self-attn용이므로 여기서는 사용하지 않음

        if self.encoder_attn is not None and encoder_out is not None:
            # ---- 1) TEXT cross-attn ----
            residual = x
            x_q = self.encoder_attn_layer_norm(x) if self.normalize_before else x

            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x_text, attn_text = self.encoder_attn(
                query=x_q,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,    
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                # need_head_weights = False
                need_head_weights=need_head_weights,
            ) # T_dec, B, C , B, T_dec, T_enc
            x_text = self.dropout_module(x_text)
            x_text = self.residual_connection(x_text, residual)
            if not self.normalize_before:
                x_text = self.encoder_attn_layer_norm(x_text)

        # ---- 2) VISION cross-attn ----
        if self.encoder_attn_vis is not None and vision_out is not None:

            x_vis, attn_vision = self.encoder_attn_vis(
                query=x_q,
                key=vision_out,
                value=vision_out,
                key_padding_mask=vision_padding_mask,   
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                # need_head_weights = False
                need_head_weights=need_head_weights,
            ) # T_dec, B, C , B, T_dec, T_enc
            x_vis = self.dropout_module(x_vis)
            x_vis = self.residual_connection(x_vis, residual)
            if not self.normalize_before:
                x_vis = self.encoder_attn_vis_layer_norm(x_vis)

        p_text   = attn_text.softmax(dim=-1)     # B, T_dec, T_enc
        p_vision = attn_vision.softmax(dim=-1)   # B, T_dec, T_enc

        eps = 1e-12
        # clamp_min(eps) prevents 0log(0) situation. 
        H_text = -(p_text.clamp_min(eps) * (p_text.clamp_min(eps)).log()).sum(dim=-1)     # B, T_dec 
        H_vis  = -(p_vision.clamp_min(eps) * (p_vision.clamp_min(eps)).log()).sum(dim=-1) # B, T_dec
        
        T_enc = p_text.size(-1)
        H_max = torch.log(torch.tensor(T_enc, device=H_text.device, dtype=H_text.dtype))
        
        H_text_n = (H_text / H_max).clamp(0, 1)  # (B, T_dec)
        H_vis_n  = (H_vis  / H_max).clamp(0, 1)

        C_text = 1.0 - H_text_n  # B, T_dec
        C_vis  = 1.0 - H_vis_n      # B, T_dec

        lambda_raw = C_vis / (C_text + C_vis + 1e-8)
        lambda_val = lambda_raw.clamp(0.0, 1.0)

        # lambda: (B, T_dec) -> (T_dec, B, 1)
        lambda_tb1 = lambda_val.transpose(0,1).unsqueeze(-1)    # T_dec, B, 1
        x = (1.0 - lambda_tb1) * x_text + lambda_tb1 * x_vis  # T_dec, B, C

        # 0.5 sum
        # x = 0.5 * x_text + 0.5 * x_vis

        x = x_text + x_vis

        ''' 
        # lambda 사용
        lamda = torch.sigmoid(self.lambda_param)
        x = lamda * x_text + (1-lamda) * x_vis
        '''

        if attn_text is not None and attn_vision is not None:
            attn = torch.cat([attn_text, attn_vision], dim=-1)
        else:
            attn = attn_text if attn_vision is None else attn_vision

        '''
        if (x_text is not None) and (x_vis is not None):
            x_fused = torch.cat([x_text, x_vis], dim=-1)   
            x = self.fuse_proj(x_fused)                     
        '''
        '''---------------------------- FFN & Norm ------------------------------'''   
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x) 

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state

        '''
        residual_text = x_text
        residual_vis = x_vis

        if self.normalize_before:
            x_text = self.final_layer_norm(x_text)
            x_vis = self.final_layer_norm(x_vis) 

        x_text = self.activation_fn(self.fc1(x_text))
        x_vis = self.activation_fn(self.fc1(x_vis))

        x_text = self.activation_dropout_module(x_text)
        x_vis = self.activation_dropout_module(x_vis)

        x_text = self.fc2(x_text)
        x_vis = self.fc2(x_vis)

        x_text = self.dropout_module(x_text)
        x_vis = self.dropout_module(x_vis)

        x_text = self.residual_connection(x_text, residual_text)
        x_vis = self.residual_connection(x_vis, residual_vis)

        x = (1.0 - lambda_tb1) * x_text + lambda_tb1 * x_vis  # T_dec, B, C

        if not self.normalize_before:
            x_text = self.final_layer_norm(x_text)
            x_vis = self.final_layer_norm(x_vis)
            # lamda = torch.sigmoid(self.lambda_param)
            x = (1.0 - lambda_tb1) * x_text + lambda_tb1 * x_vis  # T_dec, B, C
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            lamda = torch.sigmoid(self.lambda_param)
            x = (1.0 - lambda_tb1) * x_text + lambda_tb1 * x_vis  # T_dec, B, C
            return x, attn, self_attn_state
        '''

        return x, attn, None


    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

