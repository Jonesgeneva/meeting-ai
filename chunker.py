import os
import math
from pydub import AudioSegment


def convert_video_to_audio(video_path, output_path):
    """
    Converts any video file (MP4, MOV, AVI) to MP3 audio.
    Uses ffmpeg which is already installed.
    """
    print(f"🎬 Converting video to audio...")
    os.system(f'ffmpeg -i "{video_path}" -vn -acodec mp3 -q:a 2 "{output_path}" -y -loglevel quiet')

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Video converted to audio ({round(size_mb, 2)} MB)")
        return output_path
    else:
        raise Exception("❌ Video to audio conversion failed. Check ffmpeg is installed.")


def split_audio_into_chunks(audio_path, chunk_minutes=5):
    """
    Splits any audio file into equal chunks of chunk_minutes length.
    Returns list of chunk file paths.

    Example:
    30 min audio + chunk_minutes=5 → 6 chunk files
    45 min audio + chunk_minutes=5 → 9 chunk files
    """
    print(f"✂️ Loading audio file...")
    audio = AudioSegment.from_file(audio_path)

    # total duration in milliseconds
    total_ms = len(audio)
    total_minutes = total_ms / 60000
    print(f"📊 Audio duration: {round(total_minutes, 1)} minutes")

    # chunk size in milliseconds
    chunk_ms = chunk_minutes * 60 * 1000

    # calculate how many chunks we need
    total_chunks = math.ceil(total_ms / chunk_ms)
    print(f"✂️ Splitting into {total_chunks} chunks of {chunk_minutes} mins each...\n")

    # create chunks folder
    chunks_folder = os.path.join(os.path.dirname(audio_path), "chunks")
    os.makedirs(chunks_folder, exist_ok=True)

    # delete any old chunks from previous run
    for old_file in os.listdir(chunks_folder):
        os.remove(os.path.join(chunks_folder, old_file))

    chunk_paths = []

    for i in range(total_chunks):
        start_ms = i * chunk_ms
        end_ms = min((i + 1) * chunk_ms, total_ms)  # don't go beyond audio length

        chunk = audio[start_ms:end_ms]

        # name each chunk clearly
        start_min = round(start_ms / 60000, 1)
        end_min = round(end_ms / 60000, 1)
        chunk_filename = f"chunk_{i+1:02d}_{start_min}min_to_{end_min}min.mp3"
        chunk_path = os.path.join(chunks_folder, chunk_filename)

        # export chunk as MP3
        chunk.export(chunk_path, format="mp3")
        chunk_size_kb = os.path.getsize(chunk_path) / 1024
        print(f"   ✅ Chunk {i+1}/{total_chunks}: {start_min} min → {end_min} min ({round(chunk_size_kb)}KB)")
        chunk_paths.append(chunk_path)

    print(f"\n✅ All {total_chunks} chunks created successfully!")
    return chunk_paths


def transcribe_chunks(chunk_paths, whisper_model):
    """
    Transcribes each chunk one by one using the already loaded Whisper model.
    Joins all transcripts into one full transcript.
    Returns full transcript text and all segments with corrected timestamps.
    """
    full_transcript = ""
    all_segments = []
    time_offset = 0.0  # tracks real time position across chunks

    for i, chunk_path in enumerate(chunk_paths):
        chunk_name = os.path.basename(chunk_path)
        print(f"\n🎙️ Transcribing chunk {i+1}/{len(chunk_paths)}: {chunk_name}")

        result = whisper_model.transcribe(chunk_path)

        chunk_text = result["text"].strip()
        print(f"   📝 {len(chunk_text.split())} words transcribed")

        # add chunk transcript to full transcript
        full_transcript += " " + chunk_text

        # fix timestamps — add time offset so segments reflect
        # real position in the original full audio
        for seg in result["segments"]:
            corrected_seg = {
                "text": seg["text"],
                "start": round(seg["start"] + time_offset, 2),
                "end": round(seg["end"] + time_offset, 2)
            }
            all_segments.append(corrected_seg)

        # update offset for next chunk
        # get duration of this chunk
        chunk_audio = AudioSegment.from_file(chunk_path)
        time_offset += len(chunk_audio) / 1000.0  # convert ms to seconds

    print(f"\n✅ All chunks transcribed!")
    print(f"   Total words: {len(full_transcript.split())}")
    print(f"   Total segments: {len(all_segments)}")

    return full_transcript.strip(), all_segments


def process_any_file(file_path, chunk_minutes=5):
    """
    MASTER FUNCTION — handles everything automatically.

    Give it any file:
    - Short audio (under 5 min) → transcribes directly
    - Long audio (over 5 min)   → splits into chunks → transcribes each
    - Any video file            → converts to audio → splits → transcribes

    Returns full transcript and segments ready for analyse.py and sentiment.py
    """
    import whisper

    # detect file type
    extension = os.path.splitext(file_path)[1].lower()
    video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    audio_extensions = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]

    print("=" * 50)
    print(f"📁 File received: {os.path.basename(file_path)}")
    print(f"📌 Type detected: {'VIDEO' if extension in video_extensions else 'AUDIO'}")

    # step 1 — if video, convert to audio first
    if extension in video_extensions:
        audio_path = file_path.replace(extension, "_converted.mp3")
        file_path = convert_video_to_audio(file_path, audio_path)
    elif extension in audio_extensions:
        audio_path = file_path
        print(f"✅ Audio file detected — no conversion needed")
    else:
        raise Exception(f"❌ Unsupported file type: {extension}")

    # step 2 — check duration
    audio = AudioSegment.from_file(audio_path)
    total_minutes = len(audio) / 60000
    print(f"⏱️ Total duration: {round(total_minutes, 1)} minutes")

    # step 3 — load whisper model once (reused for all chunks)
    print(f"\n🧠 Loading Whisper model...")
    model = whisper.load_model("tiny")
    print(f"✅ Whisper ready")

    # step 4 — decide: direct transcribe or chunk
    if total_minutes <= chunk_minutes:
        # short audio — transcribe directly, no chunking needed
        print(f"\n⚡ Audio is under {chunk_minutes} mins — transcribing directly...")
        result = model.transcribe(audio_path)
        full_transcript = result["text"].strip()
        all_segments = result["segments"]
        print(f"✅ Direct transcription done — {len(full_transcript.split())} words")
    else:
        # long audio — split into chunks first
        print(f"\n✂️ Audio is over {chunk_minutes} mins — splitting into chunks...")
        chunk_paths = split_audio_into_chunks(audio_path, chunk_minutes)
        full_transcript, all_segments = transcribe_chunks(chunk_paths, model)

    print("=" * 50)
    return full_transcript, all_segments


if __name__ == "__main__":
    # test with your audio file
    test_path = "/content/meeting_ai/audio_samples/test_meeting.mp3"

    if os.path.exists(test_path):
        transcript, segments = process_any_file(test_path, chunk_minutes=5)
        print(f"\n--- FIRST 200 WORDS OF TRANSCRIPT ---")
        print(" ".join(transcript.split()[:200]))
    else:
        print("❌ No test audio found")
        print("Upload an audio file to /content/meeting_ai/audio_samples/ first")
