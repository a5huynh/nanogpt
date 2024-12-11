use tiktoken_rs::{cl100k_base, CoreBPE};

use super::Tokenizer;

/// Similar to https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py
/// thie implements a GPT-4 tokenizer as a light wrapper around the RegexTokenizer
struct PretrainedGTP4Tokenizer {
    model: CoreBPE
}

impl PretrainedGTP4Tokenizer {
    pub fn new() -> Self {
        Self {
            model: cl100k_base().unwrap()
        }
    }
}

impl Tokenizer for PretrainedGTP4Tokenizer {
    fn train(&mut self, text: &str, vocab_size: usize) {
        panic!("This is a pretrained model.")
    }

    fn encode(&self, text: &str) -> Vec<super::TokenSize> {
        self.model.encode(text, Default::default())
    }

    fn decode(&self, tokens: &[super::TokenSize]) -> String {
        self.model.decode(tokens).unwrap()
    }

    fn vocab(&self) -> indexmap::IndexMap<super::TokenSize, Vec<u32>> {
        panic!("This is a pretrained model.")
    }
}



/// Taken from: https://github.com/karpathy/minbpe/blob/master/minbpe/gpt4.py#L29
/// the `merges` are already the byte sequences in their merged state.
///  so we have to recover the original pairings. We can do this by doing
///  a small BPE training run on all the tokens, in their order.
///  also see https://github.com/openai/tiktoken/issues/60
///  also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
///
fn recover_merges(mergeable_ranks: IndexMap<Vec<TokenSize>, u32>) {
    let merges = IndexMap::new();

    for (tokens, rank) in mergeable_ranks.iter() {
        // skip over raw bytes
        if tokens.len() == 1 {
            continue;
        }

        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        // recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges
}