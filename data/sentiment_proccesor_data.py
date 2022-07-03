import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from config import vocab_size, oov_tok, trunc_type, max_length

class SentimentProcessorData:
    def __init__(self, df):
        print("SentimentProcessorData: initializing processor...")
        self.df = df
        self.tokenizer = None
        self.word_index = None
        self.testing_sentences = None
        self.testing_sentiments = None
        self.training_sentiments = None
        self.training_sentences = None

        self.add_sentiments_column()
        self.df_to_array()
        self.tokenize_words()

    def add_sentiments_column(self):
        self.df['sentiments'] = self.df.rating.apply(lambda x: 0 if x in [1, 2] else 1)
        pass

    def df_to_array(self):
        print("SentimentProcessorData: converting df to array...")
        split = round(len(self.df) * 0.8)
        # Train data
        # The data from beginning of the array until to the split value that is the 80 percent of the dataframe
        train_reviews = self.df['review'][:split]
        train_sentiments = self.df['sentiments'][:split]

        # Test data
        # The 20 percent left of the dataframe to test and compare with the result of the train data
        test_reviews = self.df['review'][split:]
        test_sentiments = self.df['sentiments'][split:]

        self.training_sentences = list(map(lambda x: str(x), train_reviews))
        self.training_sentiments = list(map(lambda x: x, train_sentiments))
        self.testing_sentences = list(map(lambda x: str(x), test_reviews))
        self.testing_sentiments = list(map(lambda x: x, test_sentiments))
        pass

    def tokenize_words(self):
        print("SentimentProcessorData: tokenizing words...")
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

        self.tokenizer.fit_on_texts(self.training_sentences)
        self.word_index = self.tokenizer.word_index
        pass

    def text_to_sequence_and_pad(self):
        print("SentimentProcessorData: converting texts in sequences and pads...")
        # convert to sequence the training data of the training sentences
        sequences = self.tokenizer.texts_to_sequences(self.training_sentences)
        padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

        # convert to sequence the test data of the test sentences
        test_sentences_seq = self.tokenizer.texts_to_sequences(self.testing_sentences)
        testing_padded = pad_sequences(test_sentences_seq, maxlen=max_length)

        # convert to numpy array the lists of data
        training_labels_final = np.array(self.training_sentiments)
        testing_labels_final = np.array(self.testing_sentiments)
        return padded, training_labels_final, testing_padded, testing_labels_final

    def get_data_processed(self):
        print("SentimentProcessorData: getting the result of the process...")
        return self.text_to_sequence_and_pad()
