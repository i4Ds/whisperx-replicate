import replicate

def transcribe_audio(audio_file_path):
    """ 
    Transcribe audio using a Replicate speech-to-text model.

    Parameters
    ----------
    audio_file_path : str
        Path to the audio file to be transcribed.

    Returns
    -------
    ModelOutput
        A structured output containing:
        - transcription: str
        - segments: Any
        - load_audio_ms: float
        - transcribe_ms: float

    Notes
    -----
    Uses the kenfus/stt_test model with German language setting.
    Output format is set to SRT.
    """
    output = replicate.run(
        "kenfus/stt_test_a40:a3a36fd53bb7dadcdd3ce6d574cd13e36c496b0ff686b679a42092dcf4ec363c",
        input={
            "audio_file": audio_file_path,
            "debug": True,
            "language": "de",
            "output_format": "text",
        }
    )
    return output


if __name__ == "__main__":
    output = transcribe_audio(open("84_Brugg.flac",  mode= "rb"))
    print(output)