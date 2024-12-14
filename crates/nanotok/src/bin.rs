use std::path::PathBuf;

use clap::{Parser, Subcommand};
use nanotok::{
    tokenizers::{
        basic::BasicTokenizer, gpt4::PretrainedGTP4Tokenizer, regex::RegexTokenizer, Tokenizer,
    },
    GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN,
};
use strum_macros::{Display, EnumString};

#[derive(Clone, EnumString, Display)]
pub enum TokenizerModel {
    BasicTokenizer,
    Gpt2,
    Gpt4,
    PretrainedGpt4,
}

impl Default for TokenizerModel {
    fn default() -> Self {
        Self::BasicTokenizer
    }
}

#[derive(Subcommand)]
pub enum Commands {
    Encode {
        model_file: PathBuf,
        content: String,
    },
    /// Train model from scratch, saving the model to <models/latest.bin>
    Train {
        vocab_size: usize,
        text_file: PathBuf,
    },
}

#[derive(Parser)]
#[command(version, about)]
pub struct Args {
    #[arg(short, long, default_value_t = Default::default())]
    pub model: TokenizerModel,
    #[command(subcommand)]
    pub subcommand: Commands,
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
    let mut tokenizer: Box<dyn Tokenizer> = match args.model {
        TokenizerModel::BasicTokenizer => Box::new(BasicTokenizer::new()),
        TokenizerModel::Gpt2 => Box::new(RegexTokenizer::new(GPT2_SPLIT_PATTERN)),
        TokenizerModel::Gpt4 => Box::new(RegexTokenizer::new(GPT4_SPLIT_PATTERN)),
        TokenizerModel::PretrainedGpt4 => Box::new(PretrainedGTP4Tokenizer::new()),
    };

    match args.subcommand {
        Commands::Encode {
            model_file,
            content,
        } => {
            log::info!("Loading <{}> using {model_file:?}", args.model);

            tokenizer.load(model_file)?;
            let encoded = tokenizer.encode(&content);
            println!("Encoded: {encoded:?}");
        }
        Commands::Train {
            text_file,
            vocab_size,
        } => {
            log::info!(
                "Training <{}> using {text_file:?} w/ vocab of {vocab_size}",
                args.model
            );
            let text = std::fs::read_to_string(text_file)?;

            log::info!("Training w/ {} chars", text.len());
            tokenizer.train(&text, vocab_size);

            log::info!("=== Trained Vocab ===");
            for (token_id, bytes) in tokenizer.vocab() {
                let converted = bytes.iter().map(|x| *x as u8).collect::<Vec<u8>>();
                let lossy_rep = String::from_utf8_lossy(&converted);
                println!("Token {token_id}: {bytes:?} - \"{lossy_rep}\"");
            }

            // Save model to models/tokenizers/latest_{model}.json
            let path = format!("./models/tokenizers/latest_{}.json", args.model);
            tokenizer.save(path.into())?;
        }
    }

    Ok(())
}
