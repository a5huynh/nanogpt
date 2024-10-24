use clap::{Parser, Subcommand};

#[derive(Subcommand)]
pub enum Commands {
    /// Generate text based on a trained model. Defaults to <models/latest.bin>
    Generate {
        #[arg(short, long)]
        prompt: Option<String>,
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
    #[command(subcommand)]
    pub subcommand: Option<Commands>,
}
