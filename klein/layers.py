# layers.py

import torch
from torch import Tensor
import torch.nn.functional as F

from comfy.ldm.flux.math import attention
from comfy.ldm.flux.layers import apply_mod

from ..utils import nag


class NAGKleinDoubleStreamBlock:
    """
    NAG wrapper for Klein's DoubleStreamBlock
    """
    @staticmethod
    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec,
        pe: Tensor,
        pe_negative: Tensor = None,
        attn_mask=None,
        context_pad_len: int = 0,
        nag_pad_len: int = 0,
        transformer_options=None,
        modulation_dims_img=None,
        modulation_dims_txt=None,
        **kwargs,
    ):
        """
        Modified forward for Klein double block with NAG support.
        """
        if transformer_options is None:
            transformer_options = {}

        # Get modulation
        if self.modulation:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        img_bsz = img.shape[0]
        txt_bsz = txt.shape[0]
        origin_bsz = txt_bsz - img_bsz
        
        # ===== Prepare image for attention =====
        img_modulated = self.img_norm1(img)
        img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_modulated)
        del img_modulated
        
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del img_qkv
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # ===== Prepare txt for attention =====
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        del txt_modulated
        
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del txt_qkv
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # ===== Split positive and negative for NAG =====
        if origin_bsz > 0:
            # Text splits - these are views
            txt_q_positive = txt_q[:-origin_bsz, :, context_pad_len:]
            txt_q_negative = txt_q[-origin_bsz:, :, nag_pad_len:]
            txt_k_positive = txt_k[:-origin_bsz, :, context_pad_len:]
            txt_k_negative = txt_k[-origin_bsz:, :, nag_pad_len:]
            txt_v_positive = txt_v[:-origin_bsz, :, context_pad_len:]
            txt_v_negative = txt_v[-origin_bsz:, :, nag_pad_len:]

            # Create img tensors with matching batch size for negative path
            # For positive: use all img batches (img_bsz)
            # For negative: slice to match origin_bsz
            if origin_bsz <= img_bsz:
                # Take the last origin_bsz batches of img
                img_q_neg = img_q[-origin_bsz:]
                img_k_neg = img_k[-origin_bsz:]
                img_v_neg = img_v[-origin_bsz:]
            else:
                # Need to expand img to match origin_bsz (edge case)
                repeat_times = (origin_bsz + img_bsz - 1) // img_bsz
                img_q_neg = img_q.repeat(repeat_times, 1, 1, 1)[-origin_bsz:]
                img_k_neg = img_k.repeat(repeat_times, 1, 1, 1)[-origin_bsz:]
                img_v_neg = img_v.repeat(repeat_times, 1, 1, 1)[-origin_bsz:]
        else:
            # No negative path
            txt_q_positive = txt_q[:, :, context_pad_len:]
            txt_k_positive = txt_k[:, :, context_pad_len:]
            txt_v_positive = txt_v[:, :, context_pad_len:]
            txt_q_negative = None
            txt_k_negative = None
            txt_v_negative = None
            img_q_neg = None
            img_k_neg = None
            img_v_neg = None

        # ===== Run attention for positive path =====
        if self.flipped_img_txt:
            q_pos = torch.cat((img_q, txt_q_positive), dim=2)
            k_pos = torch.cat((img_k, txt_k_positive), dim=2)
            v_pos = torch.cat((img_v, txt_v_positive), dim=2)
        else:
            q_pos = torch.cat((txt_q_positive, img_q), dim=2)
            k_pos = torch.cat((txt_k_positive, img_k), dim=2)
            v_pos = torch.cat((txt_v_positive, img_v), dim=2)

        del txt_q_positive, txt_k_positive, txt_v_positive
        
        attn_pos = attention(q_pos, k_pos, v_pos, pe=pe, mask=attn_mask, transformer_options=transformer_options)
        del q_pos, k_pos, v_pos

        # ===== Run attention for negative path (if exists) =====
        if origin_bsz > 0 and txt_q_negative is not None:
            if self.flipped_img_txt:
                q_neg = torch.cat((img_q_neg, txt_q_negative), dim=2)
                k_neg = torch.cat((img_k_neg, txt_k_negative), dim=2)
                v_neg = torch.cat((img_v_neg, txt_v_negative), dim=2)
            else:
                q_neg = torch.cat((txt_q_negative, img_q_neg), dim=2)
                k_neg = torch.cat((txt_k_negative, img_k_neg), dim=2)
                v_neg = torch.cat((txt_v_negative, img_v_neg), dim=2)

            del txt_q_negative, txt_k_negative, txt_v_negative
            del img_q_neg, img_k_neg, img_v_neg
            
            attn_neg = attention(q_neg, k_neg, v_neg, pe=pe_negative, mask=attn_mask, transformer_options=transformer_options)
            del q_neg, k_neg, v_neg
        else:
            attn_neg = None

        # Clean up original tensors
        del img_q, img_k, img_v, txt_q, txt_k, txt_v

        # Split attention outputs
        if self.flipped_img_txt:
            img_attn_pos = attn_pos[:, :img.shape[1]]
            txt_attn_pos = attn_pos[:, img.shape[1]:]
            if attn_neg is not None:
                # Note: for negative, img portion size depends on origin_bsz slice
                img_seq_len = img.shape[1]
                img_attn_neg = attn_neg[:, :img_seq_len]
                txt_attn_neg = attn_neg[:, img_seq_len:]
            else:
                img_attn_neg = None
                txt_attn_neg = None
        else:
            txt_seq_pos = attn_pos.shape[1] - img.shape[1]
            txt_attn_pos = attn_pos[:, :txt_seq_pos]
            img_attn_pos = attn_pos[:, txt_seq_pos:]
            if attn_neg is not None:
                txt_seq_neg = attn_neg.shape[1] - img.shape[1]
                txt_attn_neg = attn_neg[:, :txt_seq_neg]
                img_attn_neg = attn_neg[:, txt_seq_neg:]
            else:
                img_attn_neg = None
                txt_attn_neg = None

        del attn_pos
        if attn_neg is not None:
            del attn_neg

        # ===== Apply NAG to image attention =====
        if img_attn_neg is not None:
            if origin_bsz == img_bsz:
                # Ideal case - direct NAG
                img_attn_guided = nag(
                    img_attn_pos, 
                    img_attn_neg, 
                    self.nag_scale, 
                    self.nag_tau, 
                    self.nag_alpha
                )
            elif origin_bsz < img_bsz:
                # Apply NAG only to the last origin_bsz batches, keep others unchanged
                img_attn_guided_part = nag(
                    img_attn_pos[-origin_bsz:], 
                    img_attn_neg, 
                    self.nag_scale, 
                    self.nag_tau, 
                    self.nag_alpha
                )
                img_attn_guided = torch.cat([img_attn_pos[:-origin_bsz], img_attn_guided_part], dim=0)
                del img_attn_guided_part
            else:
                # origin_bsz > img_bsz - expand img_attn_pos to match
                repeat_times = (origin_bsz + img_bsz - 1) // img_bsz
                img_attn_pos_expanded = img_attn_pos.repeat(repeat_times, 1, 1)[-origin_bsz:]
                img_attn_guided = nag(
                    img_attn_pos_expanded, 
                    img_attn_neg, 
                    self.nag_scale, 
                    self.nag_tau, 
                    self.nag_alpha
                )
                # Trim back to img_bsz
                img_attn_guided = img_attn_guided[:img_bsz]
                del img_attn_pos_expanded
        else:
            img_attn_guided = img_attn_pos
            
        del img_attn_pos
        if img_attn_neg is not None:
            del img_attn_neg

        # ===== Calculate img blocks =====
        img_proj = self.img_attn.proj(img_attn_guided)
        del img_attn_guided
        
        img = img + apply_mod(img_proj, img_mod1.gate, None, modulation_dims_img)
        del img_proj
        
        img_mlp_in = apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)
        img_mlp_out = self.img_mlp(img_mlp_in)
        del img_mlp_in
        
        img = img + apply_mod(img_mlp_out, img_mod2.gate, None, modulation_dims_img)
        del img_mlp_out

        # ===== Calculate txt blocks =====
        if origin_bsz > 0 and txt_attn_neg is not None:
            txt_proj_pos = self.txt_attn.proj(txt_attn_pos)
            txt_proj_neg = self.txt_attn.proj(txt_attn_neg)
            del txt_attn_pos, txt_attn_neg
            
            # Apply to positive part (first img_bsz batches)
            txt_update_pos = apply_mod(txt_proj_pos, txt_mod1.gate[:-origin_bsz], None, modulation_dims_txt)
            del txt_proj_pos
            txt[:-origin_bsz, context_pad_len:] = txt[:-origin_bsz, context_pad_len:] + txt_update_pos
            del txt_update_pos
            
            # Apply to negative part (last origin_bsz batches)
            txt_update_neg = apply_mod(txt_proj_neg, txt_mod1.gate[-origin_bsz:], None, modulation_dims_txt)
            del txt_proj_neg
            txt[-origin_bsz:, nag_pad_len:] = txt[-origin_bsz:, nag_pad_len:] + txt_update_neg
            del txt_update_neg
        else:
            txt_proj = self.txt_attn.proj(txt_attn_pos)
            del txt_attn_pos
            txt = txt + apply_mod(txt_proj, txt_mod1.gate, None, modulation_dims_txt)
            del txt_proj

        # MLP
        txt_mlp_in = apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)
        txt_mlp_out = self.txt_mlp(txt_mlp_in)
        del txt_mlp_in
        
        txt = txt + apply_mod(txt_mlp_out, txt_mod2.gate, None, modulation_dims_txt)
        del txt_mlp_out

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class NAGKleinSingleStreamBlock:
    """
    NAG wrapper for Klein's SingleStreamBlock
    """
    @staticmethod
    def forward(
        self,
        x: Tensor,
        vec,
        pe: Tensor,
        pe_negative: Tensor = None,
        attn_mask=None,
        txt_length: int = None,
        img_length: int = None,
        origin_bsz: int = None,
        context_pad_len: int = 0,
        nag_pad_len: int = 0,
        transformer_options=None,
        modulation_dims=None,
        **kwargs,
    ) -> Tensor:
        """
        Modified forward for Klein single block with NAG support.
        """
        if transformer_options is None:
            transformer_options = {}

        # Get modulation
        if self.modulation is not None:
            mod, _ = self.modulation(vec)
        else:
            if isinstance(vec, tuple):
                mod = vec[0]
            else:
                mod = vec

        # Calculate batch sizes if origin_bsz not provided
        if origin_bsz is None:
            origin_bsz = 0
            
        total_bsz = x.shape[0]
        pos_bsz = total_bsz - origin_bsz if origin_bsz > 0 else total_bsz

        # Apply pre_norm and modulation, then linear1
        x_normed = self.pre_norm(x)
        x_modulated = apply_mod(x_normed, (1 + mod.scale), mod.shift, modulation_dims)
        del x_normed
        
        qkv, mlp = torch.split(
            self.linear1(x_modulated), 
            [3 * self.hidden_size, self.mlp_hidden_dim_first], 
            dim=-1
        )
        del x_modulated

        # Reshape QKV
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del qkv
        q, k = self.norm(q, k, v)

        # ===== Split positive and negative for NAG =====
        if origin_bsz > 0:
            if txt_length is not None:
                # These are views when possible
                q_positive = q[:-origin_bsz, :, context_pad_len:]
                q_negative = q[-origin_bsz:, :, nag_pad_len:]
                k_positive = k[:-origin_bsz, :, context_pad_len:]
                k_negative = k[-origin_bsz:, :, nag_pad_len:]
                v_positive = v[:-origin_bsz, :, context_pad_len:]
                v_negative = v[-origin_bsz:, :, nag_pad_len:]
                
                mlp_positive = mlp[:-origin_bsz, context_pad_len:]
                mlp_negative = mlp[-origin_bsz:, nag_pad_len:]
            else:
                # These require concat
                q_positive = torch.cat([q[:-origin_bsz, :, :img_length], q[:-origin_bsz, :, img_length + context_pad_len:]], dim=2)
                q_negative = torch.cat([q[-origin_bsz:, :, :img_length], q[-origin_bsz:, :, img_length + nag_pad_len:]], dim=2)
                k_positive = torch.cat([k[:-origin_bsz, :, :img_length], k[:-origin_bsz, :, img_length + context_pad_len:]], dim=2)
                k_negative = torch.cat([k[-origin_bsz:, :, :img_length], k[-origin_bsz:, :, img_length + nag_pad_len:]], dim=2)
                v_positive = torch.cat([v[:-origin_bsz, :, :img_length], v[:-origin_bsz, :, img_length + context_pad_len:]], dim=2)
                v_negative = torch.cat([v[-origin_bsz:, :, :img_length], v[-origin_bsz:, :, img_length + nag_pad_len:]], dim=2)
                
                mlp_positive = torch.cat([mlp[:-origin_bsz, :img_length], mlp[:-origin_bsz, img_length + context_pad_len:]], dim=1)
                mlp_negative = torch.cat([mlp[-origin_bsz:, :img_length], mlp[-origin_bsz:, img_length + nag_pad_len:]], dim=1)

            del q, k, v, mlp

            # Compute attention - sequential to reduce peak memory
            attn_positive = attention(q_positive, k_positive, v_positive, pe=pe, mask=attn_mask, transformer_options=transformer_options)
            del q_positive, k_positive, v_positive
            
            attn_negative = attention(q_negative, k_negative, v_negative, pe=pe_negative, mask=attn_mask, transformer_options=transformer_options)
            del q_negative, k_negative, v_negative

            # Extract image attention for NAG
            if txt_length is not None:
                txt_len_pos = txt_length - context_pad_len
                txt_len_neg = txt_length - nag_pad_len
                img_attn_positive = attn_positive[:, txt_len_pos:]
                img_attn_negative = attn_negative[:, txt_len_neg:]
                txt_attn_positive = attn_positive[:, :txt_len_pos]
                txt_attn_negative = attn_negative[:, :txt_len_neg]
            else:
                img_attn_positive = attn_positive[:, :img_length]
                img_attn_negative = attn_negative[:, :img_length]
                txt_attn_positive = attn_positive[:, img_length:]
                txt_attn_negative = attn_negative[:, img_length:]

            del attn_positive, attn_negative

            # Apply NAG only to image attention
            pos_img_bsz = img_attn_positive.shape[0]
            neg_img_bsz = img_attn_negative.shape[0]
            
            if pos_img_bsz == neg_img_bsz:
                img_attn_guided = nag(
                    img_attn_positive,
                    img_attn_negative, 
                    self.nag_scale, 
                    self.nag_tau, 
                    self.nag_alpha
                )
            elif neg_img_bsz < pos_img_bsz:
                # Apply NAG only to the last neg_img_bsz batches
                img_attn_guided_part = nag(
                    img_attn_positive[-neg_img_bsz:],
                    img_attn_negative, 
                    self.nag_scale, 
                    self.nag_tau, 
                    self.nag_alpha
                )
                img_attn_guided = torch.cat([img_attn_positive[:-neg_img_bsz], img_attn_guided_part], dim=0)
                del img_attn_guided_part
            else:
                # neg_img_bsz > pos_img_bsz - expand positive
                repeat_times = (neg_img_bsz + pos_img_bsz - 1) // pos_img_bsz
                img_attn_pos_expanded = img_attn_positive.repeat(repeat_times, 1, 1)[-neg_img_bsz:]
                img_attn_guided = nag(
                    img_attn_pos_expanded,
                    img_attn_negative, 
                    self.nag_scale, 
                    self.nag_tau, 
                    self.nag_alpha
                )[:pos_img_bsz]
                del img_attn_pos_expanded
                
            del img_attn_positive, img_attn_negative

            # Reconstruct with guided image attention
            if txt_length is not None:
                attn_out_positive = torch.cat([txt_attn_positive, img_attn_guided], dim=1)
                # For negative, we need to match batch sizes
                if txt_attn_negative.shape[0] == img_attn_guided.shape[0]:
                    attn_out_negative = torch.cat([txt_attn_negative, img_attn_guided], dim=1)
                else:
                    # Slice or expand img_attn_guided to match
                    if txt_attn_negative.shape[0] < img_attn_guided.shape[0]:
                        attn_out_negative = torch.cat([txt_attn_negative, img_attn_guided[-txt_attn_negative.shape[0]:]], dim=1)
                    else:
                        repeat_times = (txt_attn_negative.shape[0] + img_attn_guided.shape[0] - 1) // img_attn_guided.shape[0]
                        img_guided_expanded = img_attn_guided.repeat(repeat_times, 1, 1)[:txt_attn_negative.shape[0]]
                        attn_out_negative = torch.cat([txt_attn_negative, img_guided_expanded], dim=1)
            else:
                attn_out_positive = torch.cat([img_attn_guided, txt_attn_positive], dim=1)
                if txt_attn_negative.shape[0] == img_attn_guided.shape[0]:
                    attn_out_negative = torch.cat([img_attn_guided, txt_attn_negative], dim=1)
                else:
                    if txt_attn_negative.shape[0] < img_attn_guided.shape[0]:
                        attn_out_negative = torch.cat([img_attn_guided[-txt_attn_negative.shape[0]:], txt_attn_negative], dim=1)
                    else:
                        repeat_times = (txt_attn_negative.shape[0] + img_attn_guided.shape[0] - 1) // img_attn_guided.shape[0]
                        img_guided_expanded = img_attn_guided.repeat(repeat_times, 1, 1)[:txt_attn_negative.shape[0]]
                        attn_out_negative = torch.cat([img_guided_expanded, txt_attn_negative], dim=1)

            del txt_attn_positive, txt_attn_negative, img_attn_guided

            # Apply MLP activation
            if hasattr(self, 'yak_mlp') and self.yak_mlp:
                mlp_out_positive = self.mlp_act(mlp_positive[..., self.mlp_hidden_dim_first // 2:]) * mlp_positive[..., :self.mlp_hidden_dim_first // 2]
                mlp_out_negative = self.mlp_act(mlp_negative[..., self.mlp_hidden_dim_first // 2:]) * mlp_negative[..., :self.mlp_hidden_dim_first // 2]
            else:
                mlp_out_positive = self.mlp_act(mlp_positive)
                mlp_out_negative = self.mlp_act(mlp_negative)

            del mlp_positive, mlp_negative

            # Concatenate attention and MLP, then linear2
            combined_pos = torch.cat((attn_out_positive, mlp_out_positive), dim=2)
            del attn_out_positive, mlp_out_positive
            output_positive = self.linear2(combined_pos)
            del combined_pos
            
            combined_neg = torch.cat((attn_out_negative, mlp_out_negative), dim=2)
            del attn_out_negative, mlp_out_negative
            output_negative = self.linear2(combined_neg)
            del combined_neg

            # Apply updates
            if txt_length is not None:
                update_pos = apply_mod(output_positive, mod.gate[:-origin_bsz], None, modulation_dims)
                del output_positive
                x[:-origin_bsz, context_pad_len:] = x[:-origin_bsz, context_pad_len:] + update_pos
                del update_pos
                
                update_neg = apply_mod(output_negative, mod.gate[-origin_bsz:], None, modulation_dims)
                del output_negative
                x[-origin_bsz:, nag_pad_len:] = x[-origin_bsz:, nag_pad_len:] + update_neg
                del update_neg
            else:
                gate_pos = mod.gate[:-origin_bsz]
                gate_neg = mod.gate[-origin_bsz:]
                
                update_pos_img = apply_mod(output_positive[:, :img_length], gate_pos, None, modulation_dims)
                update_pos_txt = apply_mod(output_positive[:, img_length:], gate_pos, None, modulation_dims)
                del output_positive
                
                x[:-origin_bsz, :img_length] = x[:-origin_bsz, :img_length] + update_pos_img
                x[:-origin_bsz, img_length + context_pad_len:] = x[:-origin_bsz, img_length + context_pad_len:] + update_pos_txt
                del update_pos_img, update_pos_txt
                
                update_neg_img = apply_mod(output_negative[:, :img_length], gate_neg, None, modulation_dims)
                update_neg_txt = apply_mod(output_negative[:, img_length:], gate_neg, None, modulation_dims)
                del output_negative
                
                x[-origin_bsz:, :img_length] = x[-origin_bsz:, :img_length] + update_neg_img
                x[-origin_bsz:, img_length + nag_pad_len:] = x[-origin_bsz:, img_length + nag_pad_len:] + update_neg_txt
                del update_neg_img, update_neg_txt
        else:
            # No NAG, standard forward
            del q, k, v, mlp
            # Fallback to original behavior without NAG split
            x_normed = self.pre_norm(x)
            x_modulated = apply_mod(x_normed, (1 + mod.scale), mod.shift, modulation_dims)
            qkv, mlp = torch.split(self.linear1(x_modulated), [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)
            q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k = self.norm(q, k, v)
            
            attn = attention(q, k, v, pe=pe, mask=attn_mask, transformer_options=transformer_options)
            
            if hasattr(self, 'yak_mlp') and self.yak_mlp:
                mlp_out = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
            else:
                mlp_out = self.mlp_act(mlp)
            
            output = self.linear2(torch.cat((attn, mlp_out), dim=2))
            x = x + apply_mod(output, mod.gate, None, modulation_dims)

        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        
        return x
