use candle_core::{Tensor, IndexOp};
use rand::{Rng, SeedableRng};
use rand_pcg::Lcg64Xsh32;

pub struct Dataset {
    pub training: Tensor,
    pub validation: Tensor,
    training_len: usize,
    validation_len: usize,
    rng: Lcg64Xsh32,
}

impl Dataset {
    pub fn new(data: &Tensor) -> Self {
        let data_len = data.shape().dims1().unwrap();
        let split = (0.9 * data_len as f64).trunc() as usize;

        let train_data = data.i((..split,)).expect("Unable to split training data");
        let val_data = data.i((split..,)).expect("Unable to split validation data");
        let rng = rand_pcg::Pcg32::seed_from_u64(1337);

        Self {
            training: train_data,
            validation: val_data,
            training_len: split,
            validation_len: data_len - split,
            rng,
        }
    }

    pub fn len(&self) -> usize {
        self.training_len + self.validation_len
    }

    pub fn training_len(&self) -> usize {
        self.training_len
    }

    pub fn validation_len(&self) -> usize {
        self.validation_len
    }

    /// Generate a small batch of data of inputs x and targets y
    pub fn get_batch(&mut self, batch_size: usize, block_size: usize) -> Tensor {
        let batch_range: Vec<usize> =  (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.training_len - block_size))
            .collect();

        let rows = batch_range.iter().map(|batch_start| {
            self.training.i((*batch_start..*batch_start + block_size,))
            .unwrap()
        });

        Tensor::stack(&rows.collect::<Vec<_>>(), 0).unwrap()
    }
}