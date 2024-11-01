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
        /// Will attempt to use an existing checkpoint as a starting point vs starting
        /// from scratch
        #[arg(short, long)]
        checkpoint: Option<std::path::PathBuf>,
    },
}

#[derive(Parser)]
#[command(version, about)]
pub struct Args {
    #[arg(short, long)]
    pub gpu: bool,
    /// Change which dataset is used for the vocab + training.
    #[arg(short, long)]
    pub dataset: Option<std::path::PathBuf>,
    #[command(subcommand)]
    pub subcommand: Option<Commands>,
}
