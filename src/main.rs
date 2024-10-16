use candle_core::{Device, IndexOp, Tensor};
use dataset::Dataset;

use rand::{Rng, SeedableRng};
use std::fs;

mod dataset;
mod vocab;
use vocab::Vocab;

fn main() {
    let contents = fs::read_to_string("./data/input.txt").expect("Unable to read input file");
    println!("Building vocab...");
    let vocab = Vocab::from_content(&contents);
    println!("{vocab}");
    println!("Vocab size: {}", vocab.len());

    let device = Device::Cpu;
    let mut rng = rand_pcg::Pcg32::seed_from_u64(1337);

    let data = Tensor::new(vocab.encode(&contents), &device).expect("Unable to create tensor");
    let data = data
        .to_dtype(candle_core::DType::I64)
        .expect("Unable to cast to I64");
    println!("{:?} - {:?}", data.shape(), data.dtype());

    let mut dataset = Dataset::new(&data);

    println!("Total dataset size: {}", dataset.len());
    println!("Training set size: {}", dataset.training_len());
    println!("Validation set size: {}", dataset.validation_len());

    // How many independent sequences will we process in parallel
    let batch_size = 4;
    // What is the maximum context length for predictions
    let block_size = 8;

    let x = dataset.training.i((..block_size,)).unwrap();
    let y = dataset.training.i((1..block_size + 1,)).unwrap();

    for t in 0..block_size {
        let context = x.i((..t + 1,)).unwrap();
        let target = y.get(t).unwrap();
        println!("when input is {:?} the target is {:?}", context, target);
    }

    let batch = dataset.get_batch(batch_size, block_size);
    dbg!(batch);
}
