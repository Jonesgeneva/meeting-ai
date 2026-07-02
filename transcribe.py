import whisper

def transcribe_audio(audio_path):
    print("Loading Whisper model... (downloads once, ~500MB)")
    model = whisper.load_model("small")

    print("Transcribing audio... please wait")
    result = model.transcribe(audio_path)

    transcript = result["text"]
    segments = result["segments"]

    print("\n--- TRANSCRIPT ---")
    print(transcript)

    return transcript, segments


def get_speaker_lines(segments):
    """
    Simple speaker split — labels alternate segments as SPEAKER_A / SPEAKER_B.
    Works well for 2-person meetings as a POC.
    Replace with simple-diarizer for real speaker detection.
    """
    lines = []
    for i, seg in enumerate(segments):
        speaker = "SPEAKER_A" if i % 2 == 0 else "SPEAKER_B"
        text = seg["text"].strip()
        if text:
            lines.append({
                "speaker": speaker,
                "text": text,
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2)
            })
    return lines


if __name__ == "__main__":
    import sys

    audio_path = "/content/meeting_ai/audio_samples/test_meeting.mp3"

    if not __import__("os").path.exists(audio_path):
        print(f"❌ Audio file not found at: {audio_path}")
        print("Please upload your audio file to /content/meeting_ai/audio_samples/")
        sys.exit()

    transcript, segments = transcribe_audio(audio_path)
    speaker_lines = get_speaker_lines(segments)

    print("\n--- LABELLED LINES ---")
    for line in speaker_lines:
        print(f"[{line['start']}s] {line['speaker']}: {line['text']}")
