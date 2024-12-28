use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{conv2d, seq, Activation, Module, Sequential, VarBuilder, VarMap};

const KERNEL_SIZE: usize = 5;

///
/// Some UNets feature complex blocks at each stage, but for this toy demo
/// we’ll build a minimal example that takes in a one-channel image and passes
/// it through three convolutional layers on the down path (the down_layers in the
///  diagram and code) and three on the up path, with skip connections between the
/// down and up layers.
///
pub struct BasicUnet {
    down_layers: Sequential,
    up_layers: Sequential,
}

impl BasicUnet {
    pub fn new(in_channels: usize, out_channels: usize, device: &Device) -> Result<Self> {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);

        let down = seq()
            .add(conv2d(
                in_channels,
                32,
                KERNEL_SIZE,
                Default::default(),
                var_builder.push_prefix("down_layer_0"),
            )?)
            .add(Activation::Silu)
            .add(conv2d(
                32,
                64,
                KERNEL_SIZE,
                Default::default(),
                var_builder.push_prefix("down_layer_1"),
            )?)
            .add(Activation::Silu)
            .add(conv2d(
                64,
                64,
                KERNEL_SIZE,
                Default::default(),
                var_builder.push_prefix("down_layer_2"),
            )?);

        let up = seq()
            .add(conv2d(
                64,
                64,
                KERNEL_SIZE,
                Default::default(),
                var_builder.push_prefix("up_layer_0"),
            )?)
            .add(Activation::Silu)
            .add(conv2d(
                64,
                32,
                KERNEL_SIZE,
                Default::default(),
                var_builder.push_prefix("up_layer_1"),
            )?)
            .add(Activation::Silu)
            .add(conv2d(
                32,
                out_channels,
                KERNEL_SIZE,
                Default::default(),
                var_builder.push_prefix("up_layer_2"),
            )?);

        Ok(Self {
            down_layers: down,
            up_layers: up,
        })
    }
}

impl Module for BasicUnet {
    fn forward(&self, input: &Tensor) -> Result<candle_core::Tensor> {
        let down = self.down_layers.forward(input)?;
        self.up_layers.forward(&down)
    }
}
