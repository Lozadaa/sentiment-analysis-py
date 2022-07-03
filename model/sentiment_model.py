import tensorflow as tf

class SentimentModel:
    model = None
    def __init__(self, vocab_size, embedding_dim, max_length):
        self.define_model(vocab_size, embedding_dim, max_length)
        self.compile()

    def define_model(self, vocab_size, embedding_dim, max_length):
        print("SentimentModel: Defining model...")
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        pass

    def compile(self):
        print("SentimentModel: Compiling model...")
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        pass

    def fit(self, padded, training_labels_final, testing_padded, testing_labels_final, num_epochs = 30):
        print("SentimentModel: Fitting model...")
        result = self.model.fit(padded, training_labels_final, epochs=num_epochs,
                            validation_data=(testing_padded, testing_labels_final))
        print("SentimentModel: Successfully fit executed...")
        return result