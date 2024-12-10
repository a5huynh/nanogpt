use serde::Deserialize;
use std::{
    io::Write,
    path::{Path, PathBuf},
};
use stream::TokenSample;
use tokio::sync::mpsc::{self, Sender};

use candle_core::{backend::BackendDevice, Device, Result, Tensor};
use clap::Parser;
use cli::Commands;
use dataset::Dataset;
use model::{BigramModel, Hyperparams};
use rand::SeedableRng;

mod cli;
mod dataset;
mod model;
mod stream;
mod utils;
mod vocab;
use utils::print_probs;
use vocab::Vocab;

// -- Values used in video for testing.
// pub const BATCH_SIZE: usize = 32; // B
// pub const BLOCK_SIZE: usize = 8; // T, "time" dimension.
//
// pub const NUM_EMBED: usize = 32; // C, Number of embedding dimensions
// pub const NUM_EMBED: usize = 384; // C, Number of embedding dimensions
// pub const NUM_HEADS: usize = 4;
// pub const NUM_LAYERS: usize = 4;
// ------

pub const LEARNING_RATE: f64 = 3e-4;

pub const DEFAULT_TRAINING_STEPS: usize = 5_000;
pub const EPS: f64 = 1e-5;
pub const DROPOUT: f32 = 0.2;

pub const CONFIG_FILE: &str = "config.toml";
pub const LATEST_MODEL_PATH: &str = "./models/latest.safetensors";
pub const DEFAULT_DATASET_PATH: &str = "./data/input.txt";

#[derive(Deserialize)]
struct Config {
    pub hyperparams: Hyperparams,
}

#[tokio::main]
async fn main() -> Result<()> {
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
        } else if cfg!(target_os = "windows") || cfg!(target_os = "linux") {
            Device::Cuda(candle_core::CudaDevice::new(0)?)
        } else {
            return Err(candle_core::Error::Msg("OS not supported for GPU".into()));
        }
    } else {
        Device::Cpu
    };

    let rng = rand_pcg::Pcg32::seed_from_u64(1337);
    // Load dataset & start training
    let (vocab, data) = load_dataset(args.dataset.unwrap_or(DEFAULT_DATASET_PATH.into()), &device);
    log::info!("Vocab [{} chars] | {vocab}", vocab.len());

    let mut dataset = Dataset::new(&rng, &data);
    dataset.print_stats();

    let config = Path::new(CONFIG_FILE);
    let hyperparams: Hyperparams = if config.exists() {
        let config: Config =
            toml::from_str(&std::fs::read_to_string(config).map_err(candle_core::Error::wrap)?)
                .map_err(candle_core::Error::wrap)?;
        config.hyperparams
    } else {
        Hyperparams::default()
    };

    match args.subcommand {
        Some(Commands::Generate {
            print_probs,
            prompt,
            stream,
            num_tokens,
        }) => {
            let mut model = model::BigramModel::new(&hyperparams, 0.0, &device, &rng, vocab.len());

            let latest = Path::new(LATEST_MODEL_PATH);

            if !latest.exists() {
                return Err(candle_core::Error::Msg(format!(
                    "No model detected @ {LATEST_MODEL_PATH}"
                )));
            }

            let tx = if stream.unwrap_or_default() {
                log::info!("streaming tokens...");
                let (tx, mut rx) = mpsc::channel::<TokenSample>(32);
                let vocab = vocab.clone();
                let prompt = prompt.clone();
                tokio::spawn(async move {
                    let prompt = prompt.unwrap_or_default();
                    while let Some(message) = rx.recv().await {
                        match message {
                            TokenSample::Start => {
                                print!("{prompt}");
                            }
                            TokenSample::NewSample(sample) => {
                                let token = vocab.decode(&[sample]);
                                print!("{}", token.iter().collect::<String>());
                            }
                            TokenSample::End => {
                                println!();
                                return;
                            }
                        }
                        let _ = std::io::stdout().flush();
                    }
                });

                Some(tx)
            } else {
                None
            };

            run_generation(
                &vocab,
                &mut model,
                GenerationOptions {
                    num_tokens: num_tokens.unwrap_or(256),
                    print_probs: print_probs.unwrap_or_default(),
                    prompt: prompt.clone(),
                    stream: tx,
                },
                &device,
            )
            .await
        }
        Some(Commands::Train {
            checkpoint,
            num_steps,
        }) => {
            let mut model =
                model::BigramModel::new(&hyperparams, DROPOUT, &device, &rng, vocab.len());
            if let Some(checkpoint) = checkpoint {
                log::info!("Attempting to load checkpoint {:?}", checkpoint);
                model.parameters.load(checkpoint)?;
            }

            run_training(
                &mut dataset,
                &mut model,
                num_steps.unwrap_or(DEFAULT_TRAINING_STEPS),
            )?;

            // Reload model and set dropout to 0 to test generating
            log::info!("Testing model, generating a string...");
            let mut model = model::BigramModel::new(&hyperparams, 0.0, &device, &rng, vocab.len());
            model.parameters.load(LATEST_MODEL_PATH)?;
            run_generation(
                &vocab,
                &mut model,
                GenerationOptions {
                    num_tokens: 256,
                    print_probs: false,
                    prompt: None,
                    stream: None,
                },
                &device,
            )
            .await
        }
        None => Ok(()),
    }
}

struct GenerationOptions {
    num_tokens: usize,
    print_probs: bool,
    prompt: Option<String>,
    stream: Option<Sender<TokenSample>>,
}

async fn run_generation(
    vocab: &Vocab,
    model: &mut BigramModel,
    options: GenerationOptions,
    device: &Device,
) -> Result<()> {
    log::info!("Loading model from {LATEST_MODEL_PATH}");
    model.parameters.load(LATEST_MODEL_PATH)?;

    // Use the trained model to generate some text
    log::info!("Generating");
    let ctxt = if let Some(prompt) = options.prompt {
        let decoded = vocab.encode(&prompt);
        Tensor::new(decoded.clone(), device)?.reshape((1, decoded.len()))?
    } else {
        Tensor::zeros((1, 1), candle_core::DType::U32, device)?
    };

    let (generated, probs) = model
        .generate(&ctxt, options.num_tokens, options.stream.clone())
        .await?;

    // Only decode returned output if we're not streaming the result back.
    if options.stream.is_none() {
        let generated = generated.get(0)?.to_vec1()?;
        let decoded = vocab.decode(&generated).iter().collect::<String>();
        log::info!("Generated:\n{decoded}");
        if options.print_probs {
            for prob in probs {
                print_probs(vocab, &prob);
            }
        }
    }

    Ok(())
}

fn run_training(dataset: &mut Dataset, model: &mut BigramModel, num_steps: usize) -> Result<()> {
    log::info!("starting model training...");
    model.train(dataset, num_steps)?;
    log::info!("Saving model to {LATEST_MODEL_PATH}");
    model.parameters.save(LATEST_MODEL_PATH)?;
    Ok(())
}

fn load_dataset(dataset_file: PathBuf, device: &Device) -> (Vocab, Tensor) {
    let contents = std::fs::read_to_string(dataset_file).expect("Unable to read input file");
    let vocab = Vocab::from_content(&contents);
    let data = Tensor::new(vocab.encode(&contents), device).expect("Unable to create tensor");
    let data = data
        .to_dtype(candle_core::DType::U32)
        .expect("Unable to cast to U32");
    (vocab, data)
}

#[cfg(test)]
mod test {
    use crate::{dataset::Dataset, load_dataset, model::Hyperparams, DEFAULT_DATASET_PATH};
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

        let (vocab, data) = load_dataset(DEFAULT_DATASET_PATH.into(), &device);
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

        let (_, data) = load_dataset(DEFAULT_DATASET_PATH.into(), &device);
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
    }

    #[tokio::test]
    async fn test_generation() {
        let device = Device::Cpu;
        let rng = rand_pcg::Pcg32::seed_from_u64(1337);
        let (vocab, _) = load_dataset(DEFAULT_DATASET_PATH.into(), &device);

        let hparams = Hyperparams::default();
        let mut model = super::model::BigramModel::new(&hparams, 0.0, &device, &rng, vocab.len());

        let test = Tensor::zeros((1, 1), candle_core::DType::U32, &device).unwrap();
        let (generated, _) = model.generate(&test, 10, None).await.unwrap();
        let generated = generated.i((0, ..)).unwrap().to_vec1::<u32>().unwrap();
        let decoded = vocab.decode(&generated).iter().collect::<String>();
        assert_eq!(decoded.len(), 11);
    }
}
