import pyttsx3

def text_to_speech(text):
    """
    Converts the provided text to speech.
    
    Parameters:
        text (str): The text that you want to convert to speech.
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    print("Speech conversion completed.")

# Example usage
text_input = "Hello, welcome to the text-to-speech converter!"  # User-provided text input
text_to_speech(text_input)
