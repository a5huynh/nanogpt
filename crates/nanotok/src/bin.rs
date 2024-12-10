use std::path::PathBuf;

use clap::{Subcommand, Parser};
use nanotok::tokenizers::{basic::BasicTokenizer, Tokenizer};
use strum_macros::{Display, EnumString};

#[derive(Clone, EnumString, Display)]
pub enum TokenizerModel {
    BasicTokenizer,
    Gpt2,
    Gpt4
}

impl Default for TokenizerModel {
    fn default() -> Self {
        Self::BasicTokenizer
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// Train model from scratch, saving the model to <models/latest.bin>
    Train {
        text_file: PathBuf
    }
}

#[derive(Parser)]
#[command(version, about)]
pub struct Args {
    #[arg(short, long, default_value_t = Default::default())]
    pub model: TokenizerModel,
    #[command(subcommand)]
    pub subcommand: Commands
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Default to info logging if nothing is set.
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    // Initialize stuff
    pretty_env_logger::init();
    let args = Args::parse();
    match args.subcommand {
        Commands::Train { text_file } => {
            log::info!("Training <{}> using {text_file:?}", args.model);
            let mut tokenizer = BasicTokenizer::new();
            let text = std::fs::read_to_string(text_file)?;

            log::info!("Training w/ {} chars", text.len());
            tokenizer.train(&text, 276);

            log::info!("=== Trained Vocab ===");
            for (token_id, bytes) in tokenizer.vocab() {
                let converted = bytes.iter().map(|x| *x as u8).collect::<Vec<u8>>();
                let lossy_rep = String::from_utf8_lossy(&converted);
                println!("Token {token_id}: {bytes:?} - \"{lossy_rep}\"");
            }
        }
    }

    Ok(())
}