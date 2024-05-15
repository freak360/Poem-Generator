from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="My name is Aneeb and I am just testing my voice.",
                file_path="output.wav",
                speaker_wav="aneeb.wav",
                language="en")