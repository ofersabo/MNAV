from allennlp.data.tokenizers.word_splitter import *

@WordSplitter.register("my-bert-basic-tokenizer")
class MyBertWordSplitter(WordSplitter):
    """
    The ``BasicWordSplitter`` from the BERT implementation.
    This is used to split a sentence into words.
    Then the ``BertTokenIndexer`` converts each word into wordpieces.
    """
    def __init__(self, do_lower_case: bool = True,never_split: tuple = None) -> None:
        if never_split:
            self.never_split = never_split
            self.basic_tokenizer = BertTokenizer(do_lower_case,never_split)
        else:
            never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]","[unused1]","[unused2]","[unused3]","[unused4]")
            self.never_split = never_split
            self.basic_tokenizer = BertTokenizer(do_lower_case,never_split)

    @overrides
    def split_words(self, sentence: str) -> List[Token]:
        return [Token(text) for text in self.basic_tokenizer.tokenize(sentence)]