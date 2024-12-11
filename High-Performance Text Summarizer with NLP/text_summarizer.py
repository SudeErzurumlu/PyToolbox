from transformers import BartForConditionalGeneration, BartTokenizer

class TextSummarizer:
    def __init__(self):
        """
        Initializes the summarizer with a pre-trained BART model.
        """
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    def summarize(self, text, max_length=130, min_length=30, length_penalty=2.0, num_beams=4):
        """
        Summarizes the input text with the given parameters.
        """
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(
            inputs, max_length=max_length, min_length=min_length, length_penalty=length_penalty, num_beams=num_beams
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example Usage:
# text = "Long input text here..."
# summarizer = TextSummarizer()
# print(summarizer.summarize(text))
