use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use nanotok::tokenizers::{TokenId, Tokenizer};

#[derive(Clone, Default)]
pub struct NaiveTokenizer {
    lookup: HashMap<char, TokenId>,
    vocab: Vec<char>,
}

impl std::fmt::Display for NaiveTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<{}>",
            self.vocab.iter().collect::<String>().escape_default()
        )
    }
}

impl NaiveTokenizer {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
}

impl Tokenizer for NaiveTokenizer {
    fn decode(&self, tokens: &[TokenId]) -> String {
        let mut decoded = String::new();
        for ch in tokens.iter() {
            match self.vocab.get(ch.to_owned() as usize) {
                Some(ch) => decoded.push(*ch),
                None => eprintln!("!!! {ch} out of vocab bounds"),
            }
        }

        decoded
    }

    fn encode(&self, text: &str) -> Vec<nanotok::tokenizers::TokenSize> {
        let mut encoded = Vec::new();
        for ch in text.chars() {
            match self.lookup.get(&ch) {
                Some(idx) => encoded.push(*idx),
                None => eprintln!("!!! char <{}> not in vocab", ch),
            }
        }

        encoded
    }

    fn load(&mut self, _: std::path::PathBuf) -> anyhow::Result<()> {
        panic!("No implemented for a naive tokenizer");
    }

    fn save(&self, _: std::path::PathBuf) -> anyhow::Result<()> {
        panic!("No implemented for a naive tokenizer");
    }

    fn train(&mut self, text: &str, _vocab_size: usize) {
        let mut set = HashSet::new();
        set.extend(text.chars());

        self.vocab.clear();
        self.vocab.extend(set.iter());
        self.vocab.sort();

        self.lookup.clear();
        for (idx, ch) in self.vocab.iter().enumerate() {
            self.lookup.insert(ch.to_owned(), idx as TokenId);
        }
    }

    fn vocab(&self) -> IndexMap<nanotok::tokenizers::TokenId, Vec<u32>> {
        let mut map = IndexMap::new();
        for (idx, ch) in self.vocab.iter().enumerate() {
            let mut bytes: Vec<u8> = vec![0];
            ch.encode_utf8(&mut bytes);
            map.insert(idx as TokenId, bytes.iter().map(|x| *x as u32).collect());
        }

        map
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DEFAULT_DATASET_PATH;

    #[test]
    fn test_decode() {
        let mut tokenizer = NaiveTokenizer::new();
        tokenizer.train("0123456789", 0);

        let decoded = tokenizer.decode(&[1_u32, 2_u32, 3_u32]);
        assert_eq!(decoded, "123");
    }

    #[test]
    fn test_encode() {
        let mut tokenizer = NaiveTokenizer::new();
        tokenizer.train("0123456789", 0);

        let encoded = tokenizer.encode("123");
        assert_eq!(encoded, vec![1, 2, 3]);
    }

    #[test]
    fn test_dataset_decode() -> anyhow::Result<()> {
        let mut tokenizer = NaiveTokenizer::new();

        let content = std::fs::read_to_string(DEFAULT_DATASET_PATH)?;
        tokenizer.train(&content, 0);

        let test_string = "HELLO world, test";
        let encoded = tokenizer.encode(&test_string);
        let decoded = tokenizer.decode(&encoded);

        assert_eq!(test_string, decoded);

        Ok(())
    }
}
