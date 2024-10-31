use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
};

pub type TokenSize = u32;

#[derive(Clone)]
pub struct Vocab {
    lookup: HashMap<char, TokenSize>,
    vocab: Vec<char>,
}

impl Display for Vocab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<{}>",
            self.vocab.iter().collect::<String>().escape_default()
        )
    }
}

impl Vocab {
    pub fn new(vocab: &[char]) -> Self {
        let mut lookup = HashMap::new();
        for (idx, ch) in vocab.iter().enumerate() {
            lookup.insert(ch.to_owned(), idx as TokenSize);
        }

        Self {
            lookup,
            vocab: vocab.to_vec(),
        }
    }

    pub fn from_content(content: &str) -> Self {
        let mut chars: HashSet<char> = HashSet::new();
        chars.extend(content.chars());

        let mut chars = chars.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
        chars.sort();

        Self::new(&chars)
    }

    pub fn decode(&self, content: &[TokenSize]) -> Vec<char> {
        let mut decoded = Vec::new();
        for ch in content.iter() {
            match self.vocab.get(ch.to_owned() as usize) {
                Some(ch) => decoded.push(*ch),
                None => eprintln!("!!! {ch} out of vocab bounds"),
            }
        }

        decoded
    }

    pub fn encode(&self, content: &str) -> Vec<TokenSize> {
        let mut encoded = Vec::new();
        for ch in content.chars() {
            match self.lookup.get(&ch) {
                Some(idx) => encoded.push(*idx),
                None => eprintln!("!!! char <{}> not in vocab", ch),
            }
        }

        encoded
    }

    pub fn len(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod test {
    use crate::{load_dataset, DEFAULT_DATASET_PATH};
    use candle_core::Device;

    #[test]
    fn test_decode() {
        let vocab = super::Vocab::new(&"0123456789".chars().collect::<Vec<_>>());
        let decoded = vocab.decode(&[1_u32, 2_u32, 3_u32]);
        assert_eq!(decoded.iter().collect::<String>(), "123");
    }

    #[test]
    fn test_encode() {
        let vocab = super::Vocab::new(&"0123456789".chars().collect::<Vec<_>>());
        let encoded = vocab.encode("123");
        assert_eq!(encoded, vec![1, 2, 3]);
    }

    #[test]
    fn test_dataset_decode() {
        let device = Device::Cpu;
        let (vocab, _) = load_dataset(DEFAULT_DATASET_PATH.into(), &device);

        let test_string = "HELLO world, test";
        let encoded = vocab.encode(&test_string);
        let decoded = vocab.decode(&encoded);

        assert_eq!(test_string, decoded.iter().collect::<String>());
    }
}
