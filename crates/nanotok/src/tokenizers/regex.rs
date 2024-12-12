use std::path::PathBuf;

use fancy_regex::Regex;
use indexmap::IndexMap;

use super::{merge, str_to_tokens, BytePair, TokenId, TokenSize, Tokenizer, TokenizerModel};

/// Given a list of tokens, replace any byte pairs with the replacement id and
/// return the new list.
pub fn merge_chunks(chunks: &[Vec<u32>], pair: BytePair, replacement_id: u32) -> Vec<Vec<u32>> {
    chunks
        .iter()
        .map(|chunk| merge(chunk, pair, replacement_id))
        .collect::<Vec<_>>()
}

/// Returns the most common pair in a set of chunks and the number of times that pair occurs.
fn most_common_pair(chunks: &[Vec<u32>]) -> Option<(BytePair, usize)> {
    let mut counts: IndexMap<BytePair, usize> = IndexMap::new();

    for chunk in chunks {
        for pair in chunk.windows(2) {
            let key = (pair[0], pair[1]);
            if let Some(count) = counts.get_mut(&key) {
                *count += 1;
            } else {
                counts.insert(key, 1);
            }
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

pub struct RegexTokenizer {
    pattern: Regex,
    vocab: IndexMap<u32, Vec<u32>>,
    merges: IndexMap<BytePair, u32>,
}

impl RegexTokenizer {
    pub fn new(pattern: &str) -> Self {
        // By default, the vocav size is represented by 256 (all bytes) with no merges,
        // no patterns.
        let mut vocab = IndexMap::new();
        for idx in 0..255 {
            vocab.insert(idx, vec![idx]);
        }

        Self {
            pattern: Regex::new(pattern).expect("Unable to parse regex pattern"),
            vocab,
            merges: IndexMap::new(),
        }
    }
}

impl Tokenizer for RegexTokenizer {
    /// Use bpe to train a tokenizer model.
    fn train(&mut self, text: &str, vocab_size: usize) {
        if vocab_size < 256 {
            panic!("Vocab size must be greater than 256");
        }

        // By default, the vocav size is represented by 256 (all bytes) with no merges,
        // no patterns.
        let idx: u32 = 256;
        // how many new token ids do we need?
        // subtract by existing token ids to get the number of merges we need to do.
        let num_merges = vocab_size - 256;

        // First split the text up into chunks
        let chunks = self
            .pattern
            .find_iter(text)
            .flat_map(|x| x.ok())
            .map(|x| x.as_str().to_string())
            .collect::<Vec<_>>();

        // Then we encode each chunk into tokens
        let mut chunks = chunks
            .iter()
            .map(|string| str_to_tokens(string))
            .collect::<Vec<_>>();

        // Maps byte pairs to their new index
        let mut merges: IndexMap<BytePair, u32> = IndexMap::new();
        for merge_id in 0..num_merges {
            if let Some((pair, _)) = most_common_pair(&chunks) {
                let replacement_id = idx + merge_id as u32;
                chunks = merge_chunks(&chunks, pair, replacement_id);
                merges.insert(pair, replacement_id);
            }
        }

        // Add merges to vocab
        self.merges = merges.clone();
        for (bp, idx) in merges.iter() {
            let mut p0 = self.vocab.get(&bp.0).unwrap().to_vec();
            let p1 = self.vocab.get(&bp.1).unwrap();
            p0.extend(p1);

            self.vocab.insert(*idx, p0);
        }
    }

    fn encode(&self, _text: &str) -> Vec<TokenId> {
        todo!()
    }

    fn decode(&self, _tokens: &[TokenId]) -> String {
        todo!()
    }

    fn vocab(&self) -> IndexMap<TokenSize, Vec<u32>> {
        self.vocab.clone()
    }

    fn save(&self, path: PathBuf) -> anyhow::Result<()> {
        let model = TokenizerModel {
            version: super::ModelVersion::Version1,
            pattern: Some(self.pattern.to_string()),
            vocab: self
                .vocab
                .iter()
                .map(|(token_id, bytes)| (*token_id, bytes.to_owned()))
                .collect(),
            merges: self
                .merges
                .iter()
                .map(|(byte_pair, token_id)| (*byte_pair, *token_id))
                .collect(),
        };

        std::fs::write(path, serde_json::to_string_pretty(&model)?)?;
        Ok(())
    }

    fn load(&mut self, path: PathBuf) -> anyhow::Result<()> {
        let contents = std::fs::read_to_string(path)?;
        let model = serde_json::from_str::<TokenizerModel>(&contents)?;

        let pattern = model
            .pattern
            .ok_or(anyhow::anyhow!("No pattern found in model file"))?;
        self.pattern = Regex::new(&pattern)?;

        self.vocab.clear();
        for (token_id, bytes) in model.vocab {
            self.vocab.insert(token_id, bytes);
        }

        self.merges.clear();
        for (pair, token_id) in model.merges {
            self.merges.insert(pair, token_id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tiktoken_rs::cl100k_base;

    #[test]
    fn test_against_tiktoken() {
        // Test creating a vocab with the text
        let text = "hello world!!!? (안녕하세요!) lol123 😉";
        let tokenizer = cl100k_base().unwrap();
        let encoded = tokenizer.encode(text, Default::default());
        dbg!(&encoded);

        let decoded = tokenizer.decode(encoded).unwrap();
        dbg!(&decoded);

        assert_eq!(text, decoded);
    }
}
