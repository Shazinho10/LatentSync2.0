import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

# Assuming the following files are in the same directory or accessible in the python path
from .self_attention import SelfAttention  # Make sure this import is correct
from .unet_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    UNetMidBlock3DCrossAttn,
    get_down_block,
    get_up_block,
)
from .resnet import InflatedConv3d, InflatedGroupNorm
from ..utils.util import zero_rank_log
from .utils import zero_module


logger = logging.get_logger(__name__)


# Your MelUnet and ConvBlock classes (place them here or import them)
class ConvBlock(nn.Module):
    """A Convolutional Block with Conv -> ReLU -> Conv -> ReLU -> Self-Attention"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # Assuming SelfAttention is defined elsewhere and works with (B, SeqLen, C)
        # self.self_attention = SelfAttention(out_channels)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # The self-attention part from your original code is commented out
        # as it was not fully defined in the provided snippet.
        # If you have it, you can uncomment the following lines.
        # B, C, H, W = x.size()
        # x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        # x_attended = self.self_attention(x_flat)
        # x_attended = x_attended.permute(0, 2, 1).view(B, C, H, W)
        return x # or return x_attended


class MelUnet(nn.Module):
    def __init__(self):
        super(MelUnet, self).__init__()
        self.enc1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2) # Adjusted for simplicity
        self.dec1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pooled1 = self.pool1(enc1)
        enc2 = self.enc2(pooled1)
        pooled2 = self.pool2(enc2)

        # Bottleneck
        bottleneck = self.bottleneck(pooled2)

        # Decoder
        up2 = self.up2(bottleneck)
        # Crude cropping to handle potential size mismatches, a more robust solution
        # like padding or calculating output_padding in ConvTranspose2d is recommended.
        if up2.shape[2:] != enc2.shape[2:]:
             up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        up1 = self.up1(dec2)
        if up1.shape[2:] != enc1.shape[2:]:
             up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        output = self.final_conv(dec1)
        return output


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        # ... (all the existing arguments)
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        use_inflated_groupnorm=False,
        use_motion_module=False,
        motion_module_resolutions=(1, 2, 4, 8),
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type=None,
        motion_module_kwargs={},
        add_audio_layer=False, # We can use this flag to enable our new feature
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        self.use_motion_module = use_motion_module
        self.add_audio_layer = add_audio_layer

        # NEW: Instantiate MelUnet if audio conditioning is enabled
        if self.add_audio_layer:
            self.mel_unet = MelUnet()
            # Optional: Add a projection layer if MelUnet output dimension
            # doesn't match the required cross_attention_dim
            mel_output_channels = 1 # As per your MelUnet's final_conv
            self.audio_projection = nn.Linear(mel_output_channels, cross_attention_dim)


        self.conv_in = zero_module(InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)))
        # ... (rest of the __init__ method is the same)
        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            res = 2**i
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module
                and (res in motion_module_resolutions)
                and (not motion_module_decoder_only),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                add_audio_layer=add_audio_layer,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module and motion_module_mid_block,
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                add_audio_layer=add_audio_layer,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            res = 2 ** (3 - i)
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=use_motion_module and (res in motion_module_resolutions),
                motion_module_type=motion_module_type,
                motion_module_kwargs=motion_module_kwargs,
                add_audio_layer=add_audio_layer,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )
        self.conv_act = nn.SiLU()

        self.conv_out = zero_module(InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1))

    # ... (set_attention_slice and _set_gradient_checkpointing methods are unchanged)
    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value


    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor = None,
        # NEW: Add audio_mel as an optional input
        audio_mel: Optional[torch.FloatTensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:

        # NEW: Logic to handle audio input and generate encoder_hidden_states
        if self.add_audio_layer:
            if audio_mel is None:
                raise ValueError("audio_mel must be provided when add_audio_layer is True")

            # 1. Pass the mel spectrogram through your MelUnet
            audio_features = self.mel_unet(audio_mel) # Expected shape e.g., (B, 1, H, W)

            # 2. Reshape for cross-attention
            # The expected shape for encoder_hidden_states is (B, seq_len, cross_attention_dim)
            B, C, H, W = audio_features.shape
            audio_features = audio_features.view(B, C, H * W)
            audio_features = audio_features.permute(0, 2, 1) # -> (B, H*W, C)

            # 3. Project to the required dimension
            encoder_hidden_states = self.audio_projection(audio_features)


        # The rest of the forward pass remains largely the same
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    # Crucially, pass the new encoder_hidden_states here
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
                )
            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            for i, down_block_additional_residual in enumerate(down_block_additional_residuals):
                if down_block_additional_residual.dim() == 4:
                    down_block_additional_residual = down_block_additional_residual.unsqueeze(2)
                down_block_res_samples[i+1] = down_block_res_samples[i+1] + down_block_additional_residual

        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        if mid_block_additional_residual is not None:
            if mid_block_additional_residual.dim() == 4:
                mid_block_additional_residual = mid_block_additional_residual.unsqueeze(2)
            sample = sample + mid_block_additional_residual

        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    encoder_hidden_states=encoder_hidden_states,
                )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

    # ... (load_state_dict and from_pretrained methods are unchanged)
    def load_state_dict(self, state_dict, strict=True):
        # If the loaded checkpoint's in_channels or out_channels are different from config
        if state_dict["conv_in.weight"].shape[1] != self.config.in_channels:
            del state_dict["conv_in.weight"]
            del state_dict["conv_in.bias"]
        if state_dict["conv_out.weight"].shape[0] != self.config.out_channels:
            del state_dict["conv_out.weight"]
            del state_dict["conv_out.bias"]

        # If the loaded checkpoint's cross_attention_dim is different from config
        keys_to_remove = []
        for key in state_dict:
            if "attn2.to_k." in key or "attn2.to_v." in key:
                if state_dict[key].shape[1] != self.config.cross_attention_dim:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del state_dict[key]

        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @classmethod
    def from_pretrained(cls, model_config: dict, ckpt_path: str, device="cpu"):
        unet = cls.from_config(model_config).to(device)
        if ckpt_path != "":
            zero_rank_log(logger, f"Load from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            if "global_step" in ckpt:
                zero_rank_log(logger, f"resume from global_step: {ckpt['global_step']}")
                resume_global_step = ckpt["global_step"]
            else:
                resume_global_step = 0
            unet.load_state_dict(ckpt["state_dict"], strict=False)

            del ckpt
            torch.cuda.empty_cache()
        else:
            resume_global_step = 0

        return unet, resume_global_step