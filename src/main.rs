use std::collections::HashSet;
use std::fs;

mod vocab;

fn main() {
    let contents = fs::read_to_string("./data/input.txt").expect("Unable to read input file");

    let mut chars: HashSet<char> = HashSet::new();
    chars.extend(contents.chars());

    let mut chars = chars.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
    chars.sort();

    println!("{}", chars.iter().collect::<String>());
    println!("Vocab size: {}", chars.len());
}