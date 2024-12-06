from transformers import pipeline

class SentimentClassifier:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the sentiment classifier with a pre-trained model.
        Args:
            model_name (str): Hugging Face model to use for classification.
        """
        self.classifier = pipeline("sentiment-analysis", model=model_name)

    def analyze_text(self, text):
        """
        Analyzes the sentiment of a given text.
        Args:
            text (str): The input text.
        Returns:
            dict: Sentiment and confidence score.
        """
        result = self.classifier(text)[0]
        return {"label": result["label"], "score": round(result["score"], 4)}

    def analyze_bulk(self, texts):
        """
        Analyzes the sentiment of multiple texts.
        Args:
            texts (list): A list of input texts.
        Returns:
            list: List of results for each text.
        """
        results = self.classifier(texts)
        return [{"text": text, "label": res["label"], "score": round(res["score"], 4)} for text, res in zip(texts, results)]

# Example usage:
# classifier = SentimentClassifier()
# print(classifier.analyze_text("I love this!"))
# bulk_results = classifier.analyze_bulk(["This is great!", "I hate it here.", "It's okay."])
# for result in bulk_results:
#     print(result)
