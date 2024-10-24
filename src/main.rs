use std::path::Path;

use candle_core::{backend::BackendDevice, Device, Result, Tensor};
use clap::Parser;
use cli::Commands;
use dataset::{Dataset, RngType};
use rand::SeedableRng;

mod cli;
mod dataset;
mod model;
mod utils;
mod vocab;
use vocab::Vocab;

// pub const BATCH_SIZE: usize = 32; // B
// pub const BLOCK_SIZE: usize = 8; // T, "time" dimension.
//
// pub const NUM_EMBED: usize = 32; // C, Number of embedding dimensions
// pub const NUM_HEADS: usize = 4;
// pub const NUM_LAYERS: usize = 4;

pub const BATCH_SIZE: usize = 32; // B
pub const BLOCK_SIZE: usize = 128; // T, "time" dimension.

pub const NUM_EMBED: usize = 64; // C, Number of embedding dimensions
pub const NUM_HEADS: usize = 4;
pub const NUM_LAYERS: usize = 6;

pub const LEARNING_RATE: f64 = 1e-3;

pub const DEFAULT_TRAINING_STEPS: usize = 5_000;
pub const EPS: f64 = 1e-5;
pub const DROPOUT: f32 = 0.2;

pub const LATEST_MODEL_PATH: &str = "./models/latest.safetensors";

fn main() -> Result<()> {
    // Default to info logging if nothing is set.
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    // Initialize stuff
    pretty_env_logger::init();
    let args = cli::Args::parse();

    let device = if args.gpu {
        if cfg!(target_os = "macos") {
            Device::Metal(candle_core::MetalDevice::new(0)?)
        } else if cfg!(target_os = "windows") {
            Device::Cuda(candle_core::CudaDevice::new(0)?)
        } else {
            return Err(candle_core::Error::Msg("OS not supported for GPU".into()));
        }
    } else {
        Device::Cpu
    };

    let rng = rand_pcg::Pcg32::seed_from_u64(1337);

    match &args.subcommand {
        Some(Commands::Generate { prompt, num_tokens }) => {
            let latest = Path::new(LATEST_MODEL_PATH);

            if !latest.exists() {
                return Err(candle_core::Error::Msg(format!(
                    "No model detected @ {LATEST_MODEL_PATH}"
                )));
            }

            run_generation(prompt.clone(), num_tokens.unwrap_or(256), &device, &rng)
        }
        Some(Commands::Train { num_steps }) => {
            run_training(num_steps.unwrap_or(DEFAULT_TRAINING_STEPS), &device, &rng)
        }
        None => Ok(()),
    }
}

fn run_generation(
    prompt: Option<String>,
    num_tokens: usize,
    device: &Device,
    rng: &RngType,
) -> Result<()> {
    // Load dataset & start training
    let (vocab, _) = load_dataset(device);
    log::info!("Vocab [{} chars] | {vocab}", vocab.len());

    let mut model = model::BigramModel::new(NUM_LAYERS, device, rng, vocab.len());
    log::info!("Loading model from {LATEST_MODEL_PATH}");
    model.parameters.load(LATEST_MODEL_PATH)?;

    // Use the trained model to generate some text
    log::info!("Generating");
    let ctxt = if let Some(prompt) = prompt {
        let decoded = vocab.encode(&prompt);
        Tensor::new(decoded.clone(), device)?.reshape((1, decoded.len()))?
    } else {
        Tensor::zeros((1, 1), candle_core::DType::U32, device)?
    };

    let generated = model.generate(&ctxt, num_tokens)?;
    let generated = generated.get(0)?.to_vec1()?;
    let decoded = vocab.decode(&generated).iter().collect::<String>();
    log::info!("Generated:\n{decoded}");

    Ok(())
}

fn run_training(num_steps: usize, device: &Device, rng: &RngType) -> Result<()> {
    // Load dataset & start training
    let (vocab, data) = load_dataset(device);
    log::info!("Vocab [{} chars] | {vocab}", vocab.len());

    let mut dataset = Dataset::new(rng, &data);
    dataset.print_stats();

    let mut model = model::BigramModel::new(NUM_LAYERS, device, rng, vocab.len());
    model.train(&mut dataset, num_steps)?;
    log::info!("Saving model to {LATEST_MODEL_PATH}");
    model.parameters.save(LATEST_MODEL_PATH)?;

    // Use the trained model to generate some text
    log::info!("Testing model, generating a string...");
    let start = Tensor::zeros((1, 1), candle_core::DType::U32, device)?;
    let generated = model.generate(&start, 256)?;
    let generated = generated.get(0)?.to_vec1()?;
    let decoded = vocab.decode(&generated).iter().collect::<String>();
    log::info!("Generated: {decoded}");

    Ok(())
}

fn load_dataset(device: &Device) -> (Vocab, Tensor) {
    let contents = std::fs::read_to_string("./data/input.txt").expect("Unable to read input file");
    let vocab = Vocab::from_content(&contents);
    let data = Tensor::new(vocab.encode(&contents), device).expect("Unable to create tensor");
    let data = data
        .to_dtype(candle_core::DType::U32)
        .expect("Unable to cast to U32");
    (vocab, data)
}

#[cfg(test)]
mod test {
    use crate::{dataset::Dataset, load_dataset};
    use candle_core::{Device, IndexOp, Tensor};
    use rand::{prelude::Distribution, SeedableRng};

    #[test]
    fn test_sampling() {
        let mut rng = rand_pcg::Pcg32::seed_from_u64(1337);
        let probs = vec![0., 0., 0.75, 0.];
        let dist = rand::distributions::WeightedIndex::new(&probs).unwrap();
        let next_token = dist.sample(&mut rng) as u32;
        assert_eq!(next_token, 2);
    }

    #[test]
    fn test_logit() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::new(
            &[[
                [0u32, 1u32, 2u32, 3u32],
                [4u32, 5u32, 6u32, 7u32],
                [8u32, 9u32, 10u32, 11u32],
            ]],
            &device,
        )?;

        // focus only on the last time step
        let (_, last, _) = logits.shape().dims3()?;
        let logits = logits.i((.., last - 1, ..))?; // Becomes [B, C]
        println!("{}", logits);
        Ok(())
    }

    #[test]
    fn test_dataset_loading() {
        let device = Device::Cpu;
        let rng = rand_pcg::Pcg32::seed_from_u64(1337);

        let (vocab, data) = load_dataset(&device);
        let mut dataset = Dataset::new(&rng, &data);

        let (input, target) = dataset.get_validation_batch(1, 100);

        let decoded = vocab.decode(&input.get(0).unwrap().to_vec1().unwrap());
        dbg!(decoded.iter().collect::<String>());
        let decoded = vocab.decode(&target.get(0).unwrap().to_vec1().unwrap());
        dbg!(decoded.iter().collect::<String>());
    }

    #[test]
    fn test_batching() {
        let device = Device::Cpu;
        let rng = rand_pcg::Pcg32::seed_from_u64(1337);

        let (_, data) = load_dataset(&device);
        let mut dataset = Dataset::new(&rng, &data);

        // How many independent sequences will we process in parallel
        let batch_size = 4;
        // What is the maximum context length for predictions
        let block_size = 8;

        let (input, target) = dataset.get_batch(batch_size, block_size);

        let (x, y) = input.shape().dims2().unwrap();
        assert_eq!(x, batch_size);
        assert_eq!(y, block_size);

        let (x, y) = target.shape().dims2().unwrap();
        assert_eq!(x, batch_size);
        assert_eq!(y, block_size);
        // for bidx in 0..batch_size {
        //     for t in 0..block_size {
        //         let context = input.i((bidx, ..t + 1)).unwrap();
        //         let target = target.i((bidx, t)).unwrap();
        //         println!(
        //             "when input is {:?} the target: {}",
        //             context.to_vec1::<u32>().unwrap(),
        //             target.to_vec0::<u32>().unwrap()
        //         );
        //     }
        // }
    }

    #[test]
    fn test_generation() {
        let device = Device::Cpu;
        let rng = rand_pcg::Pcg32::seed_from_u64(1337);
        let (vocab, _) = load_dataset(&device);

        let mut model = super::model::BigramModel::new(4, &device, &rng, vocab.len());

        let test = Tensor::zeros((1, 1), candle_core::DType::U32, &device).unwrap();
        let generated = model.generate(&test, 10).unwrap();
        let generated = generated.i((0, ..)).unwrap().to_vec1::<u32>().unwrap();
        let decoded = vocab.decode(&generated).iter().collect::<String>();
        assert_eq!(decoded.len(), 11);
    }
}
