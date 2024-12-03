import torch
import torchaudio
import numpy as np
import sounddevice as sd
import wave



labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
MIN_LENGTH = 16000

sd.default.device = (0,0)
new_sample_rate = 96000
duration = 3
channels = 1
if os.path.exists("recorded_audio.wav"):
        print("deleted")
        os.remove("recorded_audio.wav")

def record_audio():
    print("Recording...")
    audio_data = sd.rec(int(duration * new_sample_rate), samplerate=new_sample_rate, channels=channels, dtype=np.int16)
    sd.wait()
    print("Recording complete.")
    output_file = "recorded_audio.wav"
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio = 2 bytes
        wf.setframerate(new_sample_rate)
        wf.writeframes(audio_data.tobytes())

    #print(f"Audio saved to {output_file}")

# Load the model state dictionary
model = torch.jit.load("Speech_model_scripted.pt")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=new_sample_rate)

def get_likely_index(tensor):
    
    # Find the most likely label index for each element in the batch
    return tensor.argmax(dim=-1)



def index_to_label(index):
    if index < len(labels):
        return labels[index]
    else:
        return "Unknown"  # or handle it as needed



def load_wav(filepath):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(filepath)
    #print("Original Sample Rate:", sample_rate)
    #print("Original Waveform Shape:", waveform.shape)

    # Dynamically set the original sample rate for resampling
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000).to(device)
    waveform = transform(waveform)
    #print("Resampled Waveform Shape:", waveform.shape)

    if waveform.shape[1] < MIN_LENGTH:
        padding = MIN_LENGTH - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    #print("Padded Waveform Shape:", waveform.shape)

    # Normalize the waveform
    waveform = waveform / waveform.abs().max()
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    #print("Final Waveform Shape (after padding and normalization):", waveform.shape)

    return waveform


def predict(filepath):
    # Load and preprocess the audio file
    waveform = load_wav(filepath)
    waveform = waveform.to(device)  # Move waveform to device
    #print("Waveform Shape before Model:", waveform.shape)

    # Run inference
    with torch.no_grad():
        output = model(waveform.unsqueeze(0))  # Add batch dimension
    #print("Model Output Shape:", output.shape)

    # Get the most likely prediction index
    prediction_index = get_likely_index(output)
    #print("Predicted Index:", prediction_index)

    # Convert index to label
    predicted_label = index_to_label(prediction_index.squeeze())
    #print("Predicted Label:", predicted_label)
    return predicted_label

def get_response(predicted_word):
    responses = {
        "backward": "Taking a step back, aren't we?",
        "bed": "Time for a nap or just cozy thoughts?",
        "bird": "Tweet-tweet! A little bird told me you're here.",
        "cat": "Meow! The cat's out of the bag.",
        "dog": "Woof! Man's best friend is here.",
        "down": "Downward we go! Don't let it bring you down.",
        "eight": "The number eight is great, isn't it?",
        "five": "High five for saying five!",
        "follow": "Lead the way, I'll follow!",
        "forward": "Onward and forward! Let's keep going.",
        "four": "Four is a fantastic number!",
        "go": "Let's go!",
        "happy": "Im pretty happy too!",
        "house": "Welcome to my humble abode!",
        "learn": "Learning never stops. What's next?",
        "left": "To the left, to the left, everything you own!",
        "marvin": "Hey Marvin! Ready to save the galaxy?",
        "nine": "Nine is divine!",
        "no": "No worries, no problem.",
        "off": "Switching off for now.",
        "on": "Turning on the magic!",
        "one": "Number one! You're the best.",
        "right": "Right on! Let's keep moving.",
        "seven": "Seven is lucky. Feeling lucky?",
        "sheila": "Hello Sheila! How's it going?",
        "six": "Six is in the mix!",
        "stop": "I will Stop",
        "three": "Three is the charm!",
        "tree": "Tree-mendous! Nature is beautiful.",
        "two": "Two's company, not a crowd.",
        "up": "Up, up, and away!",
        "visual": "I see what you did there.",
        "wow": "Wow! Just wow.",
        "yes": "Affirmative!",
        "zero": "Starting from zero and building up!"
    }
    
    # Get the response for the predicted word, or a default if not found
    return responses.get(predicted_word, "Sorry, I didn't catch that!")


record_audio()



filepath = "/home/mendel/coral/SpeechCommand/recorded_audio.wav"




predicted_label = predict(filepath)

print(get_response(predicted_label))
