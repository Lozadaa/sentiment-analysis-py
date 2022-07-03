import pandas as pd

from data.sentiment_proccesor_data import SentimentProcessorData
from model.sentiment_model import SentimentModel
from data.chart_sentiment_generator import ChartSentimentGenerator
from config import vocab_size, oov_tok, trunc_type, max_length, embedding_dim

def main():
    df = pd.read_csv(r'data/dataset/amazon_baby.csv')
    padded, training_labels_final, testing_padded, testing_labels_final = SentimentProcessorData(df).get_data_processed()
    history = SentimentModel(vocab_size, embedding_dim, max_length).fit(padded, training_labels_final, testing_padded, testing_labels_final, 5)
    ChartSentimentGenerator.generate()

main()