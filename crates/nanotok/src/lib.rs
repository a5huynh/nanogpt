pub mod tokenizers;

pub const GPT2_SPLIT_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

// NOTES:
// - GPT4 ignores case when splitting (unlike GPT2's split pattern)
pub const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

#[cfg(test)]
mod tests {
    use fancy_regex::Regex;

    #[test]
    fn test_gpt2_split_patterns() {
        let re = Regex::new(super::GPT2_SPLIT_PATTERN).unwrap();
        let matches = re.find_iter("Hello world HOW'S are you?")
            .flat_map(|x| x.ok())
            .map(|x| x.as_str().to_string())
            .collect::<Vec<_>>();

        dbg!(&matches);
        assert_eq!(matches.len(), 8);
    }


    #[test]
    fn test_gpt4_split_patterns() {
        let re = Regex::new(super::GPT4_SPLIT_PATTERN).unwrap();
        let matches = re.find_iter("Hello world HOW'S are you?")
            .flat_map(|x| x.ok())
            .map(|x| x.as_str().to_string())
            .collect::<Vec<_>>();

        dbg!(&matches);
        assert_eq!(matches.len(), 7);
    }
}
