use candle_core::{IndexOp, Tensor};
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
        let data_len = data.elem_count();
        let split = (0.9 * data_len as f64).trunc() as usize;

        let train_data = data.i((..split,)).expect("Unable to split training data");
        let val_data = data.i((split..,)).expect("Unable to split validation data");
        let rng = rand_pcg::Pcg32::seed_from_u64(1337);

        Self {
            training_len: train_data.elem_count(),
            validation_len: val_data.elem_count(),
            training: train_data,
            validation: val_data,
            rng,
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.rng = rand_pcg::Pcg32::seed_from_u64(seed);
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

    /// Generate a small batch of data of (inputs, targets)
    pub fn get_batch(&mut self, batch_size: usize, block_size: usize) -> (Tensor, Tensor) {
        let batch_range: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.training_len - block_size))
            .collect();

        let inputs = batch_range.iter().map(|batch_start| {
            self.training
                .i((*batch_start..*batch_start + block_size,))
                .unwrap()
        });

        let targets = batch_range.iter().map(|batch_start| {
            let t_start = batch_start + 1;
            self.training.i((t_start..t_start + block_size,)).unwrap()
        });

        (
            Tensor::stack(&inputs.collect::<Vec<_>>(), 0).unwrap(),
            Tensor::stack(&targets.collect::<Vec<_>>(), 0).unwrap(),
        )
    }
}
