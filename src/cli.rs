use clap::{Parser, Subcommand};

#[derive(Subcommand)]
pub enum Commands {
    /// Generate text based on a trained model. Defaults to <models/latest.bin>
    Generate,
    /// Train model from scratch, saving the model to <models/latest.bin>
    Train {
        #[arg(short, long)]
        num_steps: Option<usize>,
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
