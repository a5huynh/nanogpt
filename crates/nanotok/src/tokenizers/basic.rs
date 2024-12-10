use indexmap::IndexMap;

use super::{merge, most_common_pair, str_to_tokens, BytePair, TokenSize, Tokenizer};


/// Implementation of byte-pair encoding as an excercise.
/// Ideally if you need an actually tokenizer trained on real data,
/// use something like tiktoken.
pub struct BasicTokenizer {
    // Reverse lookup token -> bytes
    vocab: IndexMap<u32, Vec<u32>>,
    merges: IndexMap<BytePair, u32>,
}

impl BasicTokenizer {
    pub fn new() -> Self {
        // By default, the vocav size is represented by 256 (all bytes) with no merges,
        // no patterns.
        let mut vocab = IndexMap::new();
        for idx in 0..255 {
            vocab.insert(idx as u32, vec![idx]);
        }

        Self {
            vocab,
            merges: IndexMap::new(),
        }
    }
}

impl Tokenizer for BasicTokenizer {
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

        let mut tokens = str_to_tokens(&text);
        // Maps byte pairs to their new index
        let mut merges: IndexMap<BytePair, u32> = IndexMap::new();

        for merge_id in 0..num_merges {
            if let Some((pair, _)) = most_common_pair(&tokens) {
                let replacement_id = idx + merge_id as u32;
                tokens = merge(&tokens, pair, replacement_id);
                merges.insert(pair, replacement_id);
            }
        }

        // Add merges to vocab
        self.merges = merges.clone();
        for (bp, idx) in merges.iter() {
            let mut p0 = self.vocab.get(&bp.0).unwrap().to_vec();
            let p1 = self.vocab.get(&bp.1).unwrap();
            p0.extend(p1);

            self.vocab.insert(
                *idx,
                p0
            );
        }
    }

    /// Given a vocabulary, encode a string to its equivalent tokens.
    fn encode(&self, text: &str) -> Vec<TokenSize> {
        let mut tokens = str_to_tokens(text);
        for (pair, token) in self.merges.iter() {
            tokens = merge(&tokens, *pair, *token);
        }

        tokens
    }

    /// Given a vocabulary, decode an array of token ids to the string representation.
    fn decode(&self, tokens: &[TokenSize]) -> String {
        let mut string: Vec<TokenSize> = Vec::new();
        for tid in tokens {
            if let Some(decoded) = self.vocab.get(tid) {
                string.extend(decoded);
            } else {
                log::warn!("Unknown token id = {tid}");
            }
        }

        let bytes = string.iter().map(|x| *x as u8).collect::<Vec<u8>>();
        String::from_utf8_lossy(&bytes).to_string()
    }

    fn vocab(&self) -> IndexMap<TokenSize, Vec<u32>> {
        self.vocab.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenizer() {
        // Test creating a vocab with the text
        let text = include_str!("../../fixtures/test_data.txt");
        let mut tokenizer = BasicTokenizer::new();
        tokenizer.train(&text, 276);

        let compressed = tokenizer.encode(&text);
        let string = tokenizer.decode(&compressed);
        assert_eq!(string, text);

        // Test using the vocab to encode & decode a different set of characters.
        let text = include_str!("../../fixtures/test_data_2.txt");
        let rencoded = tokenizer.encode(&text);
        let decoded = tokenizer.decode( &rencoded);
        assert_eq!(text, decoded);
    }

}

