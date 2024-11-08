from transformers import BartForConditionalGeneration, BartTokenizer

class Summarizer:
    def __init__(self, model_name: str = 'facebook/bart-large-cnn'):
        # Load the pre-trained BART model and tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 50, length_penalty: float = 2.0, num_beams: int = 4):
        inputs = self.tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = self.model.generate(inputs['input_ids'], 
                                          max_length=max_length, 
                                          min_length=min_length, 
                                          length_penalty=length_penalty, 
                                          num_beams=num_beams, 
                                          early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
