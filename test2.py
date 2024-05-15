import os
import sys
import boto3
from TTS.api import TTS
import streamlit as st
from contextlib import closing
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import google.generativeai as genai
import azure.cognitiveservices.speech as speechsdk
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
from azure.core.exceptions import HttpResponseError

# # Set API keys from environment variables or directly
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
# genai.configure(api_key=Google_key)
# subscription_key = Azure_Key
# service_region = "useast"
# region = "eastus"
# key = Azure_KEY
# endpoint = "https://aneeb-translate.cognitiveservices.azure.com/"
# credential = TranslatorCredential(key, service_region)
# text_translator = TextTranslationClient(endpoint=endpoint, credential=credential)

# Initialize clients
translate_client = boto3.client(service_name='translate', rgion_name='us-easet-1', use_ssl=True)
polly_client = boto3.client("polly", region_name="us-east-1")


def generate_content(option, name, context, language):
    if option == "Poems":
        response = generate_poem(name,context)
    elif option == "Historical Events":
        response = generate_historical(name,context)
    elif option == "Islamic Stories":
        response = generate_islamic(name,context)
    elif option == "Anecdotes":
        response = generate_story(name, context)

    if language == "Urdu":
        response = translate_text(response, 'en', 'ur')

    return response


def generate_poem(name, context):
    template = """
    As a skilled poet with expertise in both traditional and creative poetry, I am looking for a poem that resonates with a specific context: {context}.
The poem should start with a warm welcome to {name}, where you will write a complete line for welcoming {name} aligning with the theme or context of the poem so that {name} should feel respected and praised.
However, please ensure that {name} is not mentioned within the body of the poem itself.
Instead, {name} should only be used at the beginning for the welcoming and at the end of the poem, to impart a moral or significant message related to the context.
I appreciate your creative and thoughtful approach to crafting this poem.
    """
    prompt = PromptTemplate(template=template, input_variables=['name', 'context'])
    model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.6)
    chain = LLMChain(llm=model, prompt=prompt, verbose=True)
    response = chain.predict(name=name, context=context)
    return response



def generate_story(name, context):
    template = """
    As a skilled storyteller with expertise in both traditional and creative stories, I am looking for a storyteller that resonates with a specific context: {context}.
The story should start with a warm welcome to {name}, where you will write a complete line for welcoming {name} aligning with the theme or context of the story so that {name} should feel respected and praised.
However, please ensure that {name} is not mentioned within the body of the story itself. Instead, {name} should only be used at the beginning for the welcoming and at the end of the story, to impart a moral or significant message related to the context.
I appreciate your creative and thoughtful approach to crafting this story.
    """
    prompt = PromptTemplate(template=template, input_variables=['name', 'context'])
    model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.6)
    chain = LLMChain(llm=model, prompt=prompt, verbose=True)
    response = chain.predict(name=name, context=context)
    return response




def generate_historical(name, context):
    template = """As an expert in history, I'm seeking information on a specific historical event related to the following context: {context}. Please provide a detailed explanation of this event, ensuring the accuracy and depth of knowledge. Importantly, the narrative should begin with a welcoming message that incorporates the individual's name, {name}, where you will write a complete line for welcoming {name} aligning with the theme or context of the historical event being discussed. However, avoid using {name} directly within the explanation of the historical event itself. Conclude the explanation by addressing {name} again, using their name to underscore the moral or key takeaway from the story. Thank you."""
    prompt = PromptTemplate(template=template, input_variables=['name', 'context'])
    model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.6)
    chain = LLMChain(llm=model, prompt=prompt, verbose=True)
    response = chain.predict(name=name, context=context)
    return response


def generate_islamic(name, context):
    template = """
    As an expert well-versed in the history and teachings of Islam, I seek an explanation of an event related to {context}. Begin with a greeting that warmly welcomes {name}, tailored to the context of the Islamic event in question. It is crucial that the narrative of the event is presented with utmost accuracy and depth, reflecting your comprehensive understanding. Please ensure that {name} is not mentioned within the main explanation of the event. Conclude your response by revisiting {name}, using their name to highlight the moral or key lesson derived from the story. This approach will help personalize the learning experience and make the moral more impactful.
    """
    prompt = PromptTemplate(template=template, input_variables=['name', 'context'])
    model = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.6)
    chain = LLMChain(llm=model, prompt=prompt, verbose=True)
    response = chain.predict(name=name, context=context)
    return response

def translate_text(text, source_language="en", target_language="ur"):
    try:
        input_text_elements = [InputTextItem(text=text)]
        response = text_translator.translate(content=input_text_elements, to=[target_language], from_parameter=source_language)
        translation = response[0] if response else None

        if translation:
            for translated_text in translation.translations:
                return f"'{translated_text.text}'."
    except HttpResponseError as exception:
        return f"Error Code: {exception.error.code}, Message: {exception.error.message}"


# Azure TTS integration
def text_to_speech_to_file_urdu(subscription_key, region, text, file_name):
    # Set up the speech configuration with your subscription key and service region
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.speech_synthesis_voice_name = "ur-PK-AsadNeural"
    
    # Specify the audio output file
    audio_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
    
    # Set up the text-to-speech synthesizer with the audio output configuration to save to a file
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    # Synthesize the text
    result = speech_synthesizer.speak_text_async(text).get()
    
    # Check the result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized to the file {file_name} for text [{text}]")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))


def text_to_speech_to_file_eng(subscription_key, region, text, file_name):
    # Set up the speech configuration with your subscription key and service region
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
    
    # Specify the audio output file
    audio_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
    
    # Set up the text-to-speech synthesizer with the audio output configuration to save to a file
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    # Synthesize the text
    result = speech_synthesizer.speak_text_async(text).get()
    
    # Check the result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesized to the file {file_name} for text [{text}]")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))


def get_audio(text, language):
    output_file = "speech.mp3"
    if language == "English":
        text_to_speech_to_file_eng(subscription_key, region, text, output_file)
    elif language == "Urdu":
        text_to_speech_to_file_urdu(subscription_key, region, text, output_file)

    st.audio(output_file)


def save_audio_stream(stream, output_file):
    try:
        with open(output_file, "wb") as file:
            file.write(stream.read())
    except IOError as error:
        print(error)
        sys.exit(-1)


def main():
    st.header("Create Your Desired Content")
    name = st.text_input("Enter Name")
    language = st.radio("Select Language", ["English", "Urdu"])
    option = st.selectbox('Choose Your Content Preference', ('Anecdotes', 'Poems', 'Historical Events', 'Islamic Stories'))
    context = st.text_input("Enter the Context")

    # Voice recording upload
    user_voice = st.file_uploader("Upload your voice recording", type=["wav"])

    if st.button("Generate"):
        content = generate_content(option, name, context, language)
        with st.expander("Content"):
            st.write(content)

        # Check if the user has uploaded a voice file
        if user_voice is not None:
            # Save the user's voice recording to a file
            save_audio_stream(user_voice, "user_voice.wav")

            # Initialize the TTS with your specific model
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

            # Generate speech with the user's voice
            tts_output = "tts_output.wav" # Output file for the TTS-generated speech
            tts.tts_to_file(text=content,
                            file_path=tts_output,
                            speaker_wav="user_voice.wav",
                            language="en" if language == "English" else "hi")

            # Play the TTS-generated speech
            st.audio(tts_output)
        else:
            # Fallback to existing TTS if no voice uploaded
            get_audio(content, language)

if __name__ == "__main__":
    main()