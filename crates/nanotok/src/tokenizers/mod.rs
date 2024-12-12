use std::path::PathBuf;

// Rust std HashMap does not preserve insertion order.
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

pub mod basic;
pub mod gpt4;
pub mod regex;

pub type BytePair = (u32, u32);
pub type TokenId = u32;
pub type TokenSize = u32;

#[derive(Deserialize, Serialize)]
pub enum ModelVersion {
    Version1,
}

#[derive(Deserialize, Serialize)]
pub struct TokenizerModel {
    version: ModelVersion,
    pattern: Option<String>,
    vocab: Vec<(TokenId, Vec<u32>)>,
    merges: Vec<(BytePair, TokenId)>,
}

pub trait Tokenizer {
    fn train(&mut self, text: &str, vocab_size: usize);
    /// Given a vocabulary, encode a string to its equivalent tokens.
    fn encode(&self, text: &str) -> Vec<TokenSize>;
    /// Given a vocabulary, decode an array of token ids to the string representation.
    fn decode(&self, tokens: &[TokenId]) -> String;
    fn vocab(&self) -> IndexMap<TokenId, Vec<u32>>;

    fn save(&self, path: PathBuf) -> anyhow::Result<()>;
    fn load(&mut self, path: PathBuf) -> anyhow::Result<()>;
}

pub fn str_to_tokens(string: &str) -> Vec<u32> {
    string
        .as_bytes()
        .iter()
        .map(|x| *x as u32)
        .collect::<Vec<_>>()
}

/// Given a list of tokens, replace any byte pairs with the replacement id and
/// return the new list.
pub fn merge(tokens: &[u32], pair: BytePair, replacement_id: u32) -> Vec<u32> {
    let mut merged = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
            merged.push(replacement_id);
            i += 2;
        } else {
            merged.push(tokens[i]);
            i += 1;
        }
    }

    merged
}

/// Returns the most common pair and the number of times that pair occurs.
pub fn most_common_pair(bytes: &[u32]) -> Option<(BytePair, usize)> {
    if bytes.len() < 2 {
        return None;
    }

    let mut counts: IndexMap<BytePair, usize> = IndexMap::new();
    // Count each pair
    for pair in bytes.windows(2) {
        let key = (pair[0], pair[1]);
        if let Some(count) = counts.get_mut(&key) {
            *count += 1;
        } else {
            counts.insert(key, 1);
        }
    }

    // Return the most common pair
    let mut count_vec = counts
        .iter()
        .map(|(key, val)| (key.to_owned(), val.to_owned()))
        .collect::<Vec<_>>();

    count_vec.sort_by(|a, b| b.1.cmp(&a.1));
    count_vec.first().map(|(k, v)| (k.to_owned(), v.to_owned()))
}
