use candle_core::{backend::BackendDevice, Device, IndexOp, MetalDevice, Tensor};
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

pub const BATCH_SIZE: usize = 32; // B
pub const BLOCK_SIZE: usize = 8; // T, "time" dimension.
pub const NUM_EMBED: usize = 32; // C, Number of embedding dimensions
pub const DEFAULT_TRAINING_STEPS: usize = 5_000;

fn main() -> Result<(), candle_core::Error> {
    // Default to info logging if nothing is set.
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    // Initialize stuff
    pretty_env_logger::init();
    let args = cli::Args::parse();

    let device = if args.gpu {
        Device::Metal(MetalDevice::new(0)?)
    } else {
        Device::Cpu
    };

    let rng = rand_pcg::Pcg32::seed_from_u64(1337);

    match &args.subcommand {
        Some(Commands::Generate) => {
            println!("Generating things");
            Ok(())
        }
        Some(Commands::Train { num_steps }) => {
            run_training(num_steps.unwrap_or(DEFAULT_TRAINING_STEPS), &device, &rng)
        }
        None => Ok(()),
    }
}

fn run_training(
    num_steps: usize,
    device: &Device,
    rng: &RngType,
) -> Result<(), candle_core::Error> {
    // Load dataset & start training
    let (vocab, data) = load_dataset(device);
    log::info!("Vocab [{} chars] | {vocab}", vocab.len());

    let mut dataset = Dataset::new(rng, &data);
    dataset.print_stats();

    let mut model = model::BigramModel::new(device, rng, vocab.len());
    model.train(&mut dataset, num_steps)?;

    // Use the trained model to generate some text
    log::info!("Testing model, generating a string...");
    let start = Tensor::zeros((1, 1), candle_core::DType::U32, device)?;
    let generated = model.generate(&start, 256)?;
    let generated = generated.i((0, ..)).unwrap().to_vec1::<u32>()?;
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
    use rand::SeedableRng;

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

        let mut model = super::model::BigramModel::new(&device, &rng, vocab.len());

        let test = Tensor::zeros((1, 1), candle_core::DType::U32, &device).unwrap();
        let generated = model.generate(&test, 10).unwrap();
        let generated = generated.i((0, ..)).unwrap().to_vec1::<u32>().unwrap();
        let decoded = vocab.decode(&generated).iter().collect::<String>();
        assert_eq!(decoded.len(), 11);
    }
}
