use nanotok::tokenizers::TokenId;

#[derive(Debug)]
pub enum TokenSample {
    Start,
    NewSample(TokenId),
    End,
}
