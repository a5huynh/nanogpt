use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{AdamW, Module, Optimizer};
use clap::Parser;
use cli::Commands;
use core::f32;
use dataset::{Dataset, RngType};
use model::head::Head;
use rand::SeedableRng;

mod cli;
mod dataset;
mod model;
mod utils;
mod vocab;
use vocab::Vocab;

pub const BATCH_SIZE: usize = 32;
pub const BLOCK_SIZE: usize = 8;
// Number of embedding dimensions
pub const NUM_EMBED: usize = 32;

fn main() -> Result<(), candle_core::Error> {
    // Initialize stuff
    pretty_env_logger::init();
    let args = cli::Args::parse();

    // let device = Device::Metal(MetalDevice::new(0)?);
    let device = Device::Cpu;
    let rng = rand_pcg::Pcg32::seed_from_u64(1337);

    match &args.subcommand {
        Some(Commands::Generate) => {
            println!("Generating things");
            Ok(())
        }
        Some(Commands::Train) => run_training(&device, &rng),
        None => {

            Ok(())
        }
    }
}

fn run_training(device: &Device, rng: &RngType) -> Result<(), candle_core::Error> {
    // Load dataset & start training
    let (vocab, data) = load_dataset(device);
    log::info!("Vocab [{} chars] | {vocab}", vocab.len());

    let mut dataset = Dataset::new(rng, &data);
    dataset.print_stats();

    let mut model = model::BigramModel::new(device, rng, vocab.len());
    train_simple_bigram_model(&mut dataset, &mut model, BATCH_SIZE, BLOCK_SIZE, 10_000)?;

    // Use the trained model to generate some text
    log::info!("Testing model, generating a string...");
    let start = Tensor::zeros((1, 1), candle_core::DType::I64, device)?;
    let generated = model.generate(&start, 100)?;
    let generated = generated.i((0, ..)).unwrap().to_vec1::<i64>()?;
    let generated = generated.iter().map(|x| *x as u32).collect::<Vec<_>>();
    let decoded = vocab.decode(&generated).iter().collect::<String>();
    log::info!("Generated: {decoded}");

    Ok(())
}

fn load_dataset(device: &Device) -> (Vocab, Tensor) {
    let contents = std::fs::read_to_string("./data/input.txt").expect("Unable to read input file");
    let vocab = Vocab::from_content(&contents);
    let data = Tensor::new(vocab.encode(&contents), device).expect("Unable to create tensor");
    let data = data
        .to_dtype(candle_core::DType::I64)
        .expect("Unable to cast to I64");
    (vocab, data)
}

fn train_simple_bigram_model(
    dataset: &mut Dataset,
    model: &mut model::BigramModel,
    // How many independent sequences will we process in parallel
    batch_size: usize,
    // What is the maximum context length for predictions
    block_size: usize,
    num_steps: usize,
) -> Result<(), candle_core::Error> {
    let mut optimizer = AdamW::new_lr(model.parameters.all_vars(), 1e-3)?;

    for step in 0..num_steps {
        // sample a batch of data
        let (input, target) = dataset.get_batch(batch_size, block_size);
        // evaluate the loss
        let (_, loss) = model.train(&input, &target)?;
        // Combines loss.backward() & optimizer.step() from pytorch.
        optimizer.backward_step(&loss)?;
        let loss = loss.to_vec0::<f32>()?;
        if step % 100 == 0 {
            log::debug!("Loss = {loss} @ step {step}");
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use crate::{dataset::Dataset, load_dataset};
    use candle_core::{Device, IndexOp, Tensor};
    use rand::SeedableRng;

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
        println!("inputs: {:?}", input.shape());
        println!("targets: {:?}", target.shape());
        println!("-----");

        for bidx in 0..batch_size {
            for t in 0..block_size {
                let context = input.i((bidx, ..t + 1)).unwrap();
                let target = target.i((bidx, t)).unwrap();
                println!(
                    "when input is {:?} the target: {}",
                    context.to_vec1::<i64>().unwrap(),
                    target.to_vec0::<i64>().unwrap()
                );
            }
        }
    }

    #[test]
    fn test_generation() {
        let device = Device::Cpu;
        let rng = rand_pcg::Pcg32::seed_from_u64(1337);
        let (vocab, _) = load_dataset(&device);

        let mut model = super::model::BigramModel::new(&device, &rng, vocab.len());

        let test = Tensor::zeros((1, 1), candle_core::DType::I64, &device).unwrap();
        let generated = model.generate(&test, 10).unwrap();
        let generated = generated.i((0, ..)).unwrap().to_vec1::<i64>().unwrap();
        let generated = generated.iter().map(|x| *x as u32).collect::<Vec<_>>();
        let decoded = vocab.decode(&generated).iter().collect::<String>();
        assert_eq!(decoded.len(), 11);
    }
}
