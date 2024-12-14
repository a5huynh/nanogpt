use std::path::PathBuf;

use clap::{ArgAction, Parser, Subcommand};

#[derive(Subcommand)]
pub enum Commands {
    /// Generate text based on a trained model. Defaults to <models/latest.bin>
    Generate {
        #[arg(long, action = ArgAction::SetTrue)]
        print_probs: Option<bool>,
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(short, long, action = ArgAction::SetTrue)]
        stream: Option<bool>,
        #[arg(short, long)]
        num_tokens: Option<usize>,
    },
    /// Train model from scratch, saving the model to <models/latest.bin>
    Train {
        #[arg(short, long)]
        num_steps: Option<usize>,
        /// Change which dataset is used for training.
        #[arg(short, long)]
        dataset_path: std::path::PathBuf,
        /// Will attempt to use an existing checkpoint as a starting point vs starting
        /// from scratch
        #[arg(short, long)]
        checkpoint: Option<std::path::PathBuf>,
    },
}

#[derive(Parser)]
#[command(version, about)]
pub struct Args {
    /// Use gpu (if available).
    #[arg(short, long)]
    pub gpu: bool,
    #[arg(long)]
    pub seed: Option<u64>,
    /// Tokenizer model file. If none is specified, assumes naive character
    /// tokenization which will use the contents of the dataset as the vocab.
    #[arg(short, long)]
    pub tokenizer: Option<PathBuf>,
    #[command(subcommand)]
    pub subcommand: Commands,
}
