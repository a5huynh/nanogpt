use tiktoken_rs::{cl100k_base, CoreBPE};

use super::Tokenizer;

/// Similar to https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py
/// thie implements a GPT-4 tokenizer as a light wrapper around the RegexTokenizer
struct PretrainedGTP4Tokenizer {
    model: CoreBPE,
}

impl PretrainedGTP4Tokenizer {
    pub fn new() -> Self {
        Self {
            model: cl100k_base().unwrap(),
        }
    }
}

impl Tokenizer for PretrainedGTP4Tokenizer {
    fn train(&mut self, _text: &str, _vocab_size: usize) {
        panic!("This is a pretrained model.")
    }

    fn encode(&self, text: &str) -> Vec<super::TokenSize> {
        self.model.encode(text, Default::default())
    }

    fn decode(&self, tokens: &[super::TokenSize]) -> String {
        self.model.decode(tokens.to_vec()).unwrap()
    }

    fn vocab(&self) -> indexmap::IndexMap<super::TokenSize, Vec<u32>> {
        panic!("This is a pretrained model.")
    }

    fn save(&self, _path: std::path::PathBuf) -> anyhow::Result<()> {
        panic!("This is a pretrained model.")
    }

    fn load(&mut self, _: std::path::PathBuf) -> anyhow::Result<()> {
        // no op
        Ok(())
    }
}
