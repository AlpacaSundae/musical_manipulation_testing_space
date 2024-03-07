import os
import librosa
import argparse
import numpy as np
import soundfile as sf

def arguments():
    parser = argparse.ArgumentParser(description="This script generates a metronome based on beat detection with librosa")

    parser.add_argument("song_file", 
                        help="song to generate the metronome for")
    parser.add_argument("beat_file",
                        help="sound bite to use as the metronome")
    parser.add_argument("--out_file", 
                        default="out/metronome.wav", 
                        help="name of file to store generated metronome in")
    parser.add_argument("--offset", 
                        default=0.0, type=float, 
                        help="UNUSED: to define a padding to offset the metronome by (as a percentage of the beat length)")

    return parser.parse_args()

# TODO: offset implemented for metronome_sound but also might want somewhere down the line to add offset to the whole audio stream
# TODO: sample_rate could be tupled with beat_samples, and metronome_sound would have its sample rate as well, need to ensure the same or convert to the same
def generate_beat_metronome(beat_samples, sample_rate, metronome_sound):
    clip_length = max(beat_samples) + len(metronome_sound)
    output = np.zeros(clip_length, dtype=np.float32)

    # offset for positioning the metronome sound so that the beat occurs at the peak 
    offset = -metronome_sound.argmax()

    for index in beat_samples:
        if index+offset > 0:
            output[index+offset:index+len(metronome_sound)+offset] = metronome_sound

    return output

def main(args):
    time_series, sample_rate = librosa.load(args.song_file)

    tempo, beat_frames = librosa.beat.beat_track(y=time_series, sr=sample_rate)

    samples_per_beat = (tempo/60) * sample_rate

    print(f"estimated tempo: {tempo:.2f} bpm, @ a sample rate of {sample_rate} Hz we have {samples_per_beat:.0f} samples/beat")

    beat_samples = librosa.frames_to_samples(beat_frames)

    output = generate_beat_metronome(beat_samples, sample_rate, librosa.load(args.beat_file)[0])

    if not os.path.exists(os.path.dirname(args.out_file)):
        os.makedirs(os.path.dirname(args.out_file))
    sf.write(args.out_file, output, sample_rate)

if __name__ == "__main__":
    args = arguments()
    main(args)