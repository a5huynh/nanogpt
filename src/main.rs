use candle_core::{Device, Tensor, IndexOp};

use std::fs;

mod vocab;
use vocab::Vocab;

fn main() {
    let contents = fs::read_to_string("./data/input.txt").expect("Unable to read input file");
    println!("Building vocab...");
    let vocab = Vocab::from_content(&contents);
    println!("{vocab}");
    println!("Vocab size: {}", vocab.len());

    let device = Device::Cpu;
    let data = Tensor::new(
        vocab.encode(&contents),
        &device
    ).expect("Unable to create tensor");
    let data = data.to_dtype(candle_core::DType::I64)
        .expect("Unable to cast to I64");
    println!("{:?} - {:?}", data.shape(), data.dtype());

    let data_len = data.shape().dims1().unwrap();
    let n = (0.9 * data_len as f64).trunc() as i64;
    println!("Training set size: {n}");
    println!("Validation set size: {}", data_len as i64 - n);

    let block_size = 8;
    let test = data.i((..block_size + 1,)).unwrap();
    dbg!(test);
}