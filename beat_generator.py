import librosa
#import argparse
import numpy as np
import soundfile as sf

# TODO: set as argparse arguments
offset_percent = 0.0
filename = "media/sued.flac"
metronome_filename = "media/beat.wav"

def generate_beat_metronome(beat_samples, sample_rate, metronome_sound):
    clip_length = max(beat_samples) + len(metronome_sound)
    output = np.zeros(clip_length, dtype=np.float32)
    offset = -metronome_sound.argmax()

    for index in beat_samples:
        if index+offset > 0:
            output[index+offset:index+len(metronome_sound)+offset] = metronome_sound

    sf.write("media/metronome.wav", output, sample_rate)

def main():
    time_series, sample_rate = librosa.load(filename)

    tempo, beat_frames = librosa.beat.beat_track(y=time_series, sr=sample_rate)

    samples_per_beat = (tempo/60) * sample_rate

    print(f"estimated tempo: {tempo:.2f} bpm, @ a sample rate of {sample_rate} Hz we have {samples_per_beat:.0f} samples/beat")

    beat_samples = librosa.frames_to_samples(beat_frames)


    generate_beat_metronome(beat_samples, sample_rate, librosa.load(metronome_filename)[0])

if __name__ == "__main__":
    main()