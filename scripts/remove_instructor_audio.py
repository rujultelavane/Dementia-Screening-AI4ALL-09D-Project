from pathlib import Path
import re
import wave

project_folder = Path(__file__).resolve().parent.parent
audio_folder = project_folder / "audio_files"
transcript_folder = project_folder / "transcript_files"
new_audio_folder = project_folder / "cut_audio_files"

def clean_audio_name(audio_file):
    name = audio_file.stem
    name = re.sub(r" \(\d+\)$", "", name)
    return name

def find_transcript(audio_file, group):
    audio_name = clean_audio_name(audio_file)
    group_transcripts = transcript_folder / group

    for transcript in sorted(group_transcripts.rglob("*.cha")):
        if transcript.stem == audio_name:
            return transcript

    return None


def find_instructor_times(transcript_file):
    instructor_times = []

    with open(transcript_file, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if not line.startswith("*INV:"):
                continue

            times = re.findall(r"\x15(\d+)_(\d+)\x15", line)
            for start, end in times:
                start = int(start)
                end = int(end)

                if end > start:
                    instructor_times.append((start, end))

    return instructor_times


def combine_overlapping_times(times):
    if not times:
        return []

    times = sorted(times)
    combined = [times[0]]

    for start, end in times[1:]:
        last_start, last_end = combined[-1]

        if start <= last_end:
            combined[-1] = (last_start, max(last_end, end))
        else:
            combined.append((start, end))

    return combined


def find_parts_to_keep(instructor_times, total_milliseconds):
    parts_to_keep = []
    current_time = 0

    for start, end in instructor_times:
        start = max(0, min(start, total_milliseconds))
        end = max(0, min(end, total_milliseconds))

        if current_time < start:
            parts_to_keep.append((current_time, start))

        current_time = max(current_time, end)

    if current_time < total_milliseconds:
        parts_to_keep.append((current_time, total_milliseconds))

    return parts_to_keep


def copy_audio_part(original_audio, new_audio, start_frame, end_frame):
    original_audio.setpos(start_frame)
    frames_left = end_frame - start_frame

    while frames_left > 0:
        frames_to_copy = min(frames_left, 100000)
        audio_data = original_audio.readframes(frames_to_copy)
        new_audio.writeframes(audio_data)
        frames_left -= frames_to_copy


def remove_instructor_voice(audio_file, transcript_file, new_file):
    instructor_times = find_instructor_times(transcript_file)
    instructor_times = combine_overlapping_times(instructor_times)

    new_file.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(audio_file), "rb") as original_audio:
        frame_rate = original_audio.getframerate()
        total_frames = original_audio.getnframes()
        total_milliseconds = round(total_frames / frame_rate * 1000)
        parts_to_keep = find_parts_to_keep(instructor_times, total_milliseconds)

        with wave.open(str(new_file), "wb") as new_audio:
            new_audio.setparams(original_audio.getparams())

            for start_ms, end_ms in parts_to_keep:
                start_frame = round(start_ms / 1000 * frame_rate)
                end_frame = round(end_ms / 1000 * frame_rate)
                copy_audio_part(original_audio, new_audio, start_frame, end_frame)

    removed_seconds = sum(end - start for start, end in instructor_times) / 1000
    return removed_seconds

total_files = 0
files_changed = 0
total_removed_seconds = 0

for group in ["Control", "Dementia"]:
    group_audio_folder = audio_folder / group

    for audio_file in sorted(group_audio_folder.glob("*.wav")):
        transcript_file = find_transcript(audio_file, group)
        new_file = new_audio_folder / group / audio_file.name

        if transcript_file is None:
            print(f"!!!!! No transcript found for {audio_file.name}")
            continue

        removed_seconds = remove_instructor_voice(audio_file, transcript_file, new_file)
        print(audio_file, " ", removed_seconds)
        total_files += 1
        total_removed_seconds += removed_seconds

        if removed_seconds > 0:
            files_changed += 1

print(f"Saved new audio files in: {new_audio_folder}")
print(f"Files processed: {total_files}")
print(f"Files with instructor audio removed: {files_changed}")
print(f"Total instructor audio removed: {total_removed_seconds:.2f} seconds")
