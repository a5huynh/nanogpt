/// Rust std HashMap does not preserve insertion order.
use indexmap::IndexMap;

type BytePair = (u32, u32);

/// Implementation of byte-pair encoding as an excercise.
/// Ideally if you need an actually tokenizer trained on real data,
/// use something like tiktoken.
pub fn decode_string(string: &str) -> Vec<u8> {
    string.as_bytes().to_owned()
}

pub fn encode_string(arr: &[u8]) -> String {
    todo!()
}

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
    use super::*;

    #[test]
    fn test_with_real_text() {
        let text = r#"Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."#;
        let tokens = text
            .as_bytes()
            .iter()
            .map(|x| *x as u32)
            .collect::<Vec<_>>();

        let (mcp, count) = most_common_pair(&tokens).unwrap();
        // let encoded = String::from_utf8([mcp.0, mcp.1].to_vec());
        assert_eq!(mcp, (101, 32));
        assert_eq!(count, 20);

        let merged = merge(&tokens, mcp, 256);
        assert_eq!(merged.len(), tokens.len() - count);
    }

    #[test]
    fn test_merge() {
        let merged = merge(&[5, 6, 6, 7, 9, 1], (6, 7), 99);
        assert_eq!(merged, vec![5, 6, 99, 9, 1]);
    }

    #[test]
    fn test_common_pair() {
        let test_string = "aaacccccbbbb"
            .as_bytes()
            .iter()
            .map(|x| *x as u32)
            .collect::<Vec<_>>();
        let pair = most_common_pair(&test_string);
        assert_eq!(pair, Some(((99, 99), 4)));

        let (pair, _) = pair.unwrap();
        let pair = [pair.0 as u8, pair.1 as u8].to_vec();
        let encoded = String::from_utf8_lossy(&pair);
        assert_eq!(encoded.to_string(), "cc".to_string());
    }
}
