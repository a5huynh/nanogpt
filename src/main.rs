use candle_core::{Device, IndexOp, Tensor};
use dataset::Dataset;
use rand::SeedableRng;

use std::fs;

mod dataset;
mod model;
mod vocab;
use vocab::Vocab;

fn main() {
    let contents = fs::read_to_string("./data/input.txt").expect("Unable to read input file");
    println!("Building vocab...");
    let vocab = Vocab::from_content(&contents);
    println!("{vocab}");
    println!("Vocab size: {}", vocab.len());

    let device = Device::Cpu;
    let rng = rand_pcg::Pcg32::seed_from_u64(1337);

    let data = Tensor::new(vocab.encode(&contents), &device).expect("Unable to create tensor");
    let data = data
        .to_dtype(candle_core::DType::I64)
        .expect("Unable to cast to I64");
    println!("{:?} - {:?}", data.shape(), data.dtype());

    let mut dataset = Dataset::new(&rng, &data);
    println!("Total dataset size: {}", dataset.len());
    println!("Training set size: {}", dataset.training_len());
    println!("Validation set size: {}", dataset.validation_len());

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

    let mut model = model::BigramModel::new(&device, &rng, vocab.len());
    let (logits, loss) = model.train(&input, &target).unwrap();
    dbg!(logits.shape());
    dbg!(loss);

    let test = Tensor::zeros((1, 1), candle_core::DType::I64, &device).unwrap();
    let generated = model.generate(&test, 1).unwrap();
    dbg!(generated);
}
