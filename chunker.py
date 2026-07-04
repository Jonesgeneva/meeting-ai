import os
import math
import subprocess


def get_audio_duration_seconds(audio_path):
    """
    Gets audio duration using ffprobe (comes with ffmpeg — no pydub needed).
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            audio_path
        ],
        capture_output=True,
        text=True
    )

    import json
    info = json.loads(result.stdout)
    duration = float(info["format"]["duration"])
    return duration


def convert_video_to_audio(video_path, output_path):
    """
    Converts any video file to MP3 audio using ffmpeg directly.
    No pydub needed.
    """
    print(f"🎬 Converting video to audio...")
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn",           # no video
            "-acodec", "mp3",
            "-q:a", "2",
            output_path,
            "-y",            # overwrite if exists
            "-loglevel", "quiet"
        ],
        check=True
    )

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Video converted to audio ({round(size_mb, 2)} MB)")
        return output_path
    else:
        raise Exception("❌ Video to audio conversion failed.")


def split_audio_into_chunks(audio_path, chunk_minutes=5):
    """
    Splits audio into equal chunks using ffmpeg directly.
    No pydub needed — works on all Python versions.
    Returns list of chunk file paths.
    """
    print(f"✂️ Getting audio duration...")

    # get total duration in seconds
    total_seconds = get_audio_duration_seconds(audio_path)
    total_minutes = total_seconds / 60
    print(f"📊 Audio duration: {round(total_minutes, 1)} minutes")

    # chunk size in seconds
    chunk_seconds = chunk_minutes * 60

    # how many chunks needed
    total_chunks = math.ceil(total_seconds / chunk_seconds)
    print(f"✂️ Splitting into {total_chunks} chunks of {chunk_minutes} mins each...\n")

    # create chunks folder next to audio file
    chunks_folder = os.path.join(os.path.dirname(audio_path), "chunks")
    os.makedirs(chunks_folder, exist_ok=True)

    # clean old chunks
    for old_file in os.listdir(chunks_folder):
        os.remove(os.path.join(chunks_folder, old_file))

    chunk_paths = []

    for i in range(total_chunks):
        start_sec = i * chunk_seconds
        end_sec = min((i + 1) * chunk_seconds, total_seconds)
        duration_sec = end_sec - start_sec

        start_min = round(start_sec / 60, 1)
        end_min = round(end_sec / 60, 1)

        chunk_filename = f"chunk_{i+1:02d}_{start_min}min_to_{end_min}min.mp3"
        chunk_path = os.path.join(chunks_folder, chunk_filename)

        # use ffmpeg to cut the chunk
        subprocess.run(
            [
                "ffmpeg",
                "-i", audio_path,
                "-ss", str(start_sec),      # start time
                "-t", str(duration_sec),    # duration
                "-acodec", "mp3",
                "-q:a", "2",
                chunk_path,
                "-y",
                "-loglevel", "quiet"
            ],
            check=True
        )

        chunk_size_kb = os.path.getsize(chunk_path) / 1024
        print(f"   ✅ Chunk {i+1}/{total_chunks}: {start_min} min → {end_min} min ({round(chunk_size_kb)} KB)")
        chunk_paths.append(chunk_path)

    print(f"\n✅ All {total_chunks} chunks created!")
    return chunk_paths


def transcribe_chunks(chunk_paths, whisper_model):
    """
    Transcribes each chunk one by one.
    Joins all transcripts with corrected timestamps.
    """
    full_transcript = ""
    all_segments = []
    time_offset = 0.0

    for i, chunk_path in enumerate(chunk_paths):
        chunk_name = os.path.basename(chunk_path)
        print(f"\n🎙️ Transcribing chunk {i+1}/{len(chunk_paths)}: {chunk_name}")

        result = whisper_model.transcribe(chunk_path)

        chunk_text = result["text"].strip()
        print(f"   📝 {len(chunk_text.split())} words transcribed")

        full_transcript += " " + chunk_text

        # fix timestamps to reflect real position in full audio
        for seg in result["segments"]:
            corrected_seg = {
                "text": seg["text"],
                "start": round(seg["start"] + time_offset, 2),
                "end": round(seg["end"] + time_offset, 2)
            }
            all_segments.append(corrected_seg)

        # get chunk duration for offset using ffprobe
        chunk_duration = get_audio_duration_seconds(chunk_path)
        time_offset += chunk_duration

    print(f"\n✅ All chunks transcribed!")
    print(f"   Total words    : {len(full_transcript.split())}")
    print(f"   Total segments : {len(all_segments)}")

    return full_transcript.strip(), all_segments


def process_any_file(file_path, chunk_minutes=5):
    """
    MASTER FUNCTION — handles everything automatically.
    Works for any audio or video file of any length.
    No pydub — pure ffmpeg + Python.
    """
    import whisper

    extension = os.path.splitext(file_path)[1].lower()
    video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    audio_extensions = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]

    print("=" * 50)
    print(f"📁 File: {os.path.basename(file_path)}")
    print(f"📌 Type: {'VIDEO' if extension in video_extensions else 'AUDIO'}")

    # step 1 — convert video to audio if needed
    if extension in video_extensions:
        audio_path = file_path.replace(extension, "_converted.mp3")
        file_path = convert_video_to_audio(file_path, audio_path)
    elif extension in audio_extensions:
        audio_path = file_path
        print(f"✅ Audio file — no conversion needed")
    else:
        raise Exception(f"❌ Unsupported file type: {extension}")

    # step 2 — get duration
    total_seconds = get_audio_duration_seconds(audio_path)
    total_minutes = total_seconds / 60
    print(f"⏱️ Duration: {round(total_minutes, 1)} minutes")

    # step 3 — load whisper once
    print(f"\n🧠 Loading Whisper model...")
    model = whisper.load_model("tiny")
    print(f"✅ Whisper ready")

    # step 4 — direct transcribe or chunk
    if total_minutes <= chunk_minutes:
        print(f"\n⚡ Under {chunk_minutes} mins — transcribing directly...")
        result = model.transcribe(audio_path)
        full_transcript = result["text"].strip()
        all_segments = result["segments"]
        print(f"✅ Done — {len(full_transcript.split())} words")
    else:
        print(f"\n✂️ Over {chunk_minutes} mins — splitting into chunks...")
        chunk_paths = split_audio_into_chunks(audio_path, chunk_minutes)
        full_transcript, all_segments = transcribe_chunks(chunk_paths, model)

    print("=" * 50)
    return full_transcript, all_segments
