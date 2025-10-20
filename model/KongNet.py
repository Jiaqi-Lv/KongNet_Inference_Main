import segmentation_models_pytorch as smp
import segmentation_models_pytorch.base.initialization as init
import timm
from typing import List, Optional, Union, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.base import modules as md
from torchvision.ops import Conv2dNormActivation

class TimmEncoderFixed(nn.Module):
    """Fixed version of TIMM encoder that handles drop_path_rate parameter properly.
    
    This encoder wraps TIMM models to provide consistent feature extraction interface
    for segmentation tasks. It extracts features at multiple scales from the encoder
    backbone.
    
    Args:
        name (str): Name of the TIMM model to use as backbone
        pretrained (bool): Whether to use pretrained weights. Default: True
        in_channels (int): Number of input channels. Default: 3
        depth (int): Number of encoder stages to extract features from. Default: 5
        output_stride (int): Output stride of the encoder. Default: 32
        drop_rate (float): Dropout rate. Default: 0.5
        drop_path_rate (Optional[float]): Drop path rate for stochastic depth. Default: 0.0
    """
    
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        drop_rate: float = 0.5,
        drop_path_rate: Optional[float] = 0.0,
    ) -> None:
        super().__init__()
        if drop_path_rate is None:
            kwargs = dict(
                in_chans=in_channels,
                features_only=True,
                pretrained=pretrained,
                out_indices=tuple(range(depth)),
                drop_rate=drop_rate,
            )
        else:
            kwargs = dict(
                in_chans=in_channels,
                features_only=True,
                pretrained=pretrained,
                out_indices=tuple(range(depth)),
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            List[torch.Tensor]: List of feature tensors at different scales,
                including the input as the first element
        """
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self) -> List[int]:
        """Get output channels for each feature level.
        
        Returns:
            List[int]: Number of channels at each feature level
        """
        return self._out_channels

    @property
    def output_stride(self) -> int:
        """Get the output stride of the encoder.
        
        Returns:
            int: Output stride value
        """
        return min(self._output_stride, 2**self._depth)


def get_timm_encoder(
    name: str,
    in_channels: int = 3,
    depth: int = 5,
    weights: bool = False,
    output_stride: int = 32,
    drop_rate: float = 0.5,
    drop_path_rate: float = 0.25,
) -> TimmEncoderFixed:
    """Create a TIMM encoder instance.
    
    Args:
        name (str): Name of the TIMM model
        in_channels (int): Number of input channels. Default: 3
        depth (int): Encoder depth. Default: 5
        weights (bool): Whether to use pretrained weights. Default: False
        output_stride (int): Output stride. Default: 32
        drop_rate (float): Dropout rate. Default: 0.5
        drop_path_rate (float): Drop path rate. Default: 0.25
        
    Returns:
        TimmEncoderFixed: Configured encoder instance
    """
    encoder = TimmEncoderFixed(
        name,
        weights,
        in_channels,
        depth,
        output_stride,
        drop_rate,
        drop_path_rate,
    )
    return encoder


def get_KongNet(
    enc: str = "tf_efficientnetv2_l.in21k_ft_in1k",
    num_heads: int = 3,
    decoders_out_channels: List[int] = [3, 3, 3],
    use_batchnorm: bool = True,
    attention_type: str = 'scse',
    center: bool = True,
    wide_decoder: bool = False,
) -> 'KongNet':
    """Create a KongNet model with specified configuration.
    
    KongNet is a multi-head segmentation model that uses multiple decoders
    to produce different output types (e.g., different cell types).
    
    Args:
        enc (str): Encoder backbone name. Default: "tf_efficientnetv2_l.in21k_ft_in1k"
        num_heads (int): Number of decoder heads. Default: 3
        decoders_out_channels (List[int]): Output channels for each decoder head. Default: [3, 3, 3]
        use_batchnorm (bool): Whether to use batch normalization. Default: True
        attention_type (str): Type of attention mechanism. Default: 'scse'
        center (bool): Whether to use center block. Default: True
        wide_decoder (bool): Whether to use wider decoder channels. Default: False
        
    Returns:
        KongNet: Configured KongNet model instance
        
    Raises:
        ValueError: If num_heads doesn't match length of decoders_out_channels
    """
    # Bug fix: Add validation for consistent num_heads and decoders_out_channels
    if num_heads != len(decoders_out_channels):
        raise ValueError(
            f"num_heads ({num_heads}) must match length of decoders_out_channels "
            f"({len(decoders_out_channels)})"
        )
    
    if num_heads < 1 or len(decoders_out_channels) < 1:
        raise ValueError("num_heads and decoders_out_channels must be at least 1")
    
    depth = 5

    encoder = get_timm_encoder(
        name=enc,
        in_channels=3,
        depth=depth,
        output_stride=32,
        drop_rate=0.5,
        drop_path_rate=0.25,
    )

    decoder_channels = (256, 128, 64, 32, 16)[:depth]
    if wide_decoder:
        decoder_channels = (512, 256, 128, 64, 32)[:depth]

    decoders = []
    for i in range(num_heads):
        decoders.append(
            KongNetDecoder(
                encoder_channels=encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=len(decoder_channels),
                use_batchnorm=use_batchnorm,
                center=center,
                attention_type=attention_type,
            )
        )

    heads = []
    for i in range(num_heads):
        heads.append(
            smp.base.SegmentationHead(
                in_channels=decoders[i]
                .blocks[-1]
                .conv2[0]
                .out_channels,
                out_channels=decoders_out_channels[
                    i
                ],  # instance channels
                activation=None,
                kernel_size=1,
            )
        )

    model = KongNet(encoder, decoders, heads)

    return model


class Conv2dReLU(nn.Sequential):
    """2D Convolution followed by BatchNorm (optional) and ReLU activation.
    
    This is a common building block for neural networks that combines
    convolution, normalization, and activation in a single module.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel
        padding (Union[int, Tuple[int, int]]): Padding added to input. Default: 0
        stride (Union[int, Tuple[int, int]]): Stride of the convolution. Default: 1
        use_batchnorm (bool): Whether to use batch normalization. Default: True
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
        stride: Union[int, Tuple[int, int]] = 1,
        use_batchnorm: bool = True,
    ) -> None:
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU()

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SubPixelUpsample(nn.Module):
    """Sub-pixel upsampling module using PixelShuffle.
    
    This module performs upsampling using sub-pixel convolution (PixelShuffle)
    which is more efficient than transposed convolution and produces better results.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        upscale_factor (int): Factor to increase spatial resolution. Default: 2
    """
    
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int = 2) -> None:
        super(SubPixelUpsample, self).__init__()
        self.conv1 = Conv2dNormActivation(
            in_channels,
            out_channels * upscale_factor**2,
            kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.conv2 = Conv2dNormActivation(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sub-pixel upsampling.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Upsampled tensor of shape (B, out_channels, H*upscale_factor, W*upscale_factor)
        """
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and attention.
    
    This block performs upsampling of the input features, concatenates with skip connections
    from the encoder, applies attention mechanisms, and processes through convolutions.
    
    Args:
        in_channels (int): Number of input channels
        skip_channels (int): Number of channels from skip connection
        out_channels (int): Number of output channels
        attention_type (str): Type of attention mechanism. Default: 'scse'
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        attention_type: str = 'scse',
    ) -> None:
        super().__init__()
        self.up = SubPixelUpsample(
            in_channels, in_channels, upscale_factor=2
        )
        self.conv1 = Conv2dNormActivation(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dNormActivation(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.SiLU,
        )
        self.attention2 = md.Attention(
            attention_type, in_channels=out_channels
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through decoder block.
        
        Args:
            x (torch.Tensor): Input tensor to be upsampled
            skip (Optional[torch.Tensor]): Skip connection tensor from encoder. Default: None
            
        Returns:
            torch.Tensor: Processed output tensor
        """
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Module):
    """Center block that applies attention mechanism at the bottleneck.
    
    This block is placed at the center of the U-Net architecture (deepest level)
    to enhance feature representation using attention mechanisms.
    
    Args:
        in_channels (int): Number of input channels
    """
    
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.attention = md.Attention("scse", in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through center block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with attention applied
        """
        x = self.attention(x)
        return x


class KongNetDecoder(nn.Module):
    """Decoder module for KongNet architecture.
    
    This decoder implements a U-Net style decoder with multiple decoder blocks,
    attention mechanisms, and optional center block at the bottleneck.
    
    Args:
        encoder_channels (List[int]): Number of channels at each encoder level
        decoder_channels (Tuple[int, ...]): Number of channels at each decoder level
        n_blocks (int): Number of decoder blocks. Default: 5
        use_batchnorm (bool): Whether to use batch normalization. Default: True
        attention_type (str): Type of attention mechanism. Default: 'scse'
        center (bool): Whether to use center block at bottleneck. Default: True
        
    Raises:
        ValueError: If n_blocks doesn't match length of decoder_channels
    """
    
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: Tuple[int, ...],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: str = 'scse',
        center: bool = True,
    ) -> None:
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            # Bug fix: CenterBlock only takes in_channels parameter
            self.center = CenterBlock(head_channels)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        # Bug fix: DecoderBlock doesn't use use_batchnorm parameter
        kwargs = dict(attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(
                in_channels, skip_channels, out_channels
            )
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            *features: Variable number of feature tensors from encoder at different scales
            
        Returns:
            torch.Tensor: Decoded output tensor
        """
        features = features[
            1:
        ]  # remove first skip with same spatial resolution
        features = features[
            ::-1
        ]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class KongNet(torch.nn.Module):
    """KongNet: Multi-head segmentation model.
    
    KongNet is a segmentation model with multiple decoder heads that can
    produce different types of segmentation outputs simultaneously. It uses
    a shared encoder and multiple task-specific decoders.
    
    Args:
        encoder: Encoder module (e.g., TimmEncoderFixed)
        decoder_list (List[nn.Module]): List of decoder modules
        head_list (List[nn.Module]): List of segmentation heads
        
    Raises:
        ValueError: If decoder_list and head_list have different lengths
    """
    
    def __init__(
        self, 
        encoder: nn.Module, 
        decoder_list: List[nn.Module], 
        head_list: List[nn.Module]
    ) -> None:
        super(KongNet, self).__init__()
        
        # Bug fix: Add validation for matching decoder and head lists
        if len(decoder_list) != len(head_list):
            raise ValueError(
                f"Number of decoders ({len(decoder_list)}) must match "
                f"number of heads ({len(head_list)})"
            )
        
        self.encoder = encoder  # Bug fix: Simplified encoder assignment
        self.decoders = nn.ModuleList(decoder_list)
        self.heads = nn.ModuleList(head_list)
        self.initialize()

    def initialize(self) -> None:
        """Initialize decoder and head weights using segmentation_models_pytorch initializers."""
        for decoder in self.decoders:
            init.initialize_decoder(decoder)
        for head in self.heads:
            init.initialize_head(head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Concatenated output from all heads of shape (B, sum(head_channels), H, W)
        """
        features = self.encoder(x)
        decoder_outputs = []
        for decoder in self.decoders:
            decoder_outputs.append(decoder(*features))

        masks = []
        for head, decoder_output in zip(self.heads, decoder_outputs):
            masks.append(head(decoder_output))

        return torch.cat(masks, 1)