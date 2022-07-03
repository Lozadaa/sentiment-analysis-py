from matplotlib import pyplot as plt


class ChartSentimentGenerator:
    @staticmethod
    def generate(history):
        print("Generating charts...")
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'r', 'Training Accuracy')
        plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
        plt.title('Training and validation accuracy')
        plt.figure()
        plt.show()
        plt.plot(epochs, loss, 'r', 'Training Loss')
        plt.plot(epochs, val_loss, 'b', 'Validation Loss')
        plt.title('Training and validation loss')
        plt.show()
        pass