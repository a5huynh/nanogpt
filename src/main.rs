use std::collections::HashSet;
use std::fs;

fn main() {
    let contents = fs::read_to_string("./data/input.txt").expect("Unable to read input file");

    let mut chars: HashSet<char> = HashSet::new();
    chars.extend(contents.chars());

    let mut chars = chars.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
    chars.sort();

    println!("{}", chars.iter().collect::<String>());
    println!("Vocab size: {}", chars.len());
}

fn decode(vocab: &[char], content: &[usize]) -> Vec<char> {
    let mut decoded = Vec::new();
    for ch in content.iter() {
        match vocab.get(ch.to_owned()) {
            Some(ch) => decoded.push(*ch),
            None => eprintln!("!!! {ch} out of vocab bounds"),
        }
    }

    decoded
}

fn encode(vocab: &[char], content: &str) -> Vec<usize> {
    let mut encoded = Vec::new();
    for ch in content.chars() {
        if let Ok(idx) = vocab.binary_search(&ch) {
            encoded.push(idx);
        } else {
            eprintln!("!!! char <{}> not in vocab", ch);
        }
    }

    encoded
}

#[cfg(test)]
mod test {
    #[test]
    fn test_decode() {
        let vocab: Vec<char> = "0123456789".chars().collect();
        let decoded = super::decode(&vocab, &[1_usize, 2_usize, 3_usize]);
        assert_eq!(decoded.iter().collect::<String>(), "123");
    }

    #[test]
    fn test_encode() {
        let vocab: Vec<char> = "0123456789".chars().collect();
        let encoded = super::encode(&vocab, "123");
        assert_eq!(encoded, vec![1, 2, 3]);
    }
}
