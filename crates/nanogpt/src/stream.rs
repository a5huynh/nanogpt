use crate::vocab::TokenSize;

#[derive(Debug)]
pub enum TokenSample {
    Start,
    NewSample(TokenSize),
    End,
}
