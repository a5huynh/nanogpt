/// Rust std HashMap does not preserve insertion order.
use indexmap::IndexMap;

pub type BytePair = (u32, u32);

pub const GPT2_SPLIT_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

// NOTES:
// - GPT4 ignores case when splitting (unlike GPT2's split pattern)
pub const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Implementation of byte-pair encoding as an excercise.
/// Ideally if you need an actually tokenizer trained on real data,
/// use something like tiktoken.
pub fn str_to_tokens(string: &str) -> Vec<u32> {
    string
        .as_bytes()
        .iter()
        .map(|x| *x as u32)
        .collect::<Vec<_>>()
}

/// Given a vocabulary, encode a string to its equivalent tokens.
pub fn encode_to_tokens(vocab: &IndexMap<BytePair, u32>, string: &str) -> Vec<u32> {
    let mut tokens = str_to_tokens(string);
    for (pair, token) in vocab.iter() {
        tokens = merge(&tokens, *pair, *token);
    }

    tokens
}

/// Given a vocabulary, decode an array of token ids to the string representation.
pub fn tokens_to_str(vocab: &IndexMap<BytePair, u32>, tokens: &[u32]) -> String {
    let mut string = Vec::new();
    let mut tokens = tokens.to_owned();

    // Loop until we have no more tokens to replace, in reverse order
    for (pair, token) in vocab.iter().rev() {
        for tid in tokens {
            if tid == *token {
                string.push(pair.0);
                string.push(pair.1);
            } else {
                string.push(tid);
            }
        }

        tokens = string.clone();
        string.clear();
    }

    // Convert to bytes and convert to a string
    let bytes = tokens.iter().map(|x| *x as u8).collect::<Vec<u8>>();
    String::from_utf8_lossy(&bytes).to_string()
}

pub fn bpe(text: &str, vocab_size: usize) -> (Vec<u32>, IndexMap<BytePair, u32>) {
    // new token ids
    let idx: u32 = 256;
    // subtract by existing token ids to get the number of merges we need to do.
    let num_merges = vocab_size - 256;

    let mut tokens = str_to_tokens(text);
    // Maps byte pairs to their new index
    let mut merges: IndexMap<BytePair, u32> = IndexMap::new();

    for merge_id in 0..num_merges {
        if let Some((pair, _)) = most_common_pair(&tokens) {
            let replacement_id = idx + merge_id as u32;
            tokens = merge(&tokens, pair, replacement_id);
            merges.insert(pair, replacement_id);
        }
    }

    (tokens, merges)
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

#[cfg(test)]
mod tests {
    use fancy_regex::Regex;

    use super::*;

    #[test]
    fn test_with_real_text() {
        let text = include_str!("../fixtures/test_data.txt");
        let tokens = str_to_tokens(&text);

        let (mcp, count) = most_common_pair(&tokens).unwrap();
        // let encoded = String::from_utf8([mcp.0, mcp.1].to_vec());
        assert_eq!(mcp, (101, 32));
        assert_eq!(count, 638);

        // Test creating a vocab with the text
        let (compressed, vocab) = bpe(&text, 276);
        let string = tokens_to_str(&vocab, &compressed);
        assert_eq!(string, text);

        // Test using the vocab to encode & decode a different set of characters.
        let text = include_str!("../fixtures/test_data_2.txt");
        let rencoded = encode_to_tokens(&vocab, &text);
        let decoded = tokens_to_str(&vocab, &rencoded);
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_merge() {
        let merged = merge(&[5, 6, 6, 7, 9, 1], (6, 7), 99);
        assert_eq!(merged, vec![5, 6, 99, 9, 1]);
    }

    #[test]
    fn test_common_pair() {
        let tokens = str_to_tokens("aaacccccbbbb");
        let pair = most_common_pair(&tokens);
        assert_eq!(pair, Some(((99, 99), 4)));

        let (pair, _) = pair.unwrap();
        let pair = [pair.0 as u8, pair.1 as u8].to_vec();
        let encoded = String::from_utf8_lossy(&pair);
        assert_eq!(encoded.to_string(), "cc".to_string());
    }

    #[test]
    fn test_gpt2_split_patterns() {
        let re = Regex::new(super::GPT2_SPLIT_PATTERN).unwrap();
        let matches = re.find_iter("Hello world HOW'S are you?")
            .flat_map(|x| x.ok())
            .map(|x| x.as_str().to_string())
            .collect::<Vec<_>>();

        dbg!(&matches);
        assert_eq!(matches.len(), 20);
    }


    #[test]
    fn test_gpt4_split_patterns() {
        let re = Regex::new(super::GPT4_SPLIT_PATTERN).unwrap();
        let matches = re.find_iter("Hello world HOW'S are you?")
            .flat_map(|x| x.ok())
            .map(|x| x.as_str().to_string())
            .collect::<Vec<_>>();

        dbg!(&matches);
        assert_eq!(matches.len(), 20);
    }
}
