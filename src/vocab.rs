use std::collections::HashMap;

pub struct Vocab {
    lookup: HashMap<char, usize>,
    vocab: Vec<char>,
}

impl Vocab {
    pub fn new(vocab: &[char]) -> Self {
        let mut lookup = HashMap::new();
        for (idx, ch) in vocab.iter().enumerate() {
            lookup.insert(ch.to_owned(), idx);
        }

        Self {
            lookup,
            vocab: vocab.to_vec(),
        }
    }

    pub fn decode(&self, content: &[usize]) -> Vec<char> {
        let mut decoded = Vec::new();
        for ch in content.iter() {
            match self.vocab.get(ch.to_owned()) {
                Some(ch) => decoded.push(*ch),
                None => eprintln!("!!! {ch} out of vocab bounds"),
            }
        }

        decoded
    }

    pub fn encode(&self, content: &str) -> Vec<usize> {
        let mut encoded = Vec::new();
        for ch in content.chars() {
            match self.lookup.get(&ch) {
                Some(idx) => encoded.push(*idx),
                None => eprintln!("!!! char <{}> not in vocab", ch),
            }
        }

        encoded
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_decode() {
        let vocab = super::Vocab::new(&"0123456789".chars().collect::<Vec<_>>());
        let decoded = vocab.decode(&[1_usize, 2_usize, 3_usize]);
        assert_eq!(decoded.iter().collect::<String>(), "123");
    }

    #[test]
    fn test_encode() {
        let vocab = super::Vocab::new(&"0123456789".chars().collect::<Vec<_>>());
        let encoded = vocab.encode("123");
        assert_eq!(encoded, vec![1, 2, 3]);
    }
}
