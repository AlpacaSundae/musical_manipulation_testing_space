#
#  Experimenting with a theoretical method of storing an audio timeline
#

class Timeline:
    """
    This class manages the timeline for a single track
    
    Class Variables
    ---------------
    timeline : dict --> number, TimelineBlock
      * this holds the timeline for a single audio track
      * audio segments are stored with their start index and the audio block (TimelineBlock class) in a dictionary

    block_list : list of TimelineBlock
      * the block list mainly exists so that blocks can be destroyed from the timeline but still be later readded
      * potential for an area in the UI which is a bin of the currently defined blocks
    """
    def __init__(self) -> None:
        self.timeline = {}
        self.block_list = []

    def new_block(self, audio_handler):
        self.block_list.append(TimelineBlock(audio_handler))

    def insert_block(self, idx: int, block):
        # if there is a block at this index, we want to instead move its start position to trim after the new block
        # i.e. inserting a new block is destructive to the overlapped previous block
        if idx in self.timeline.keys():
            move_block = self.timeline[idx]
            new_idx = idx + block.len()

            # potential error case where the start idx is beyond the length of the block
            # in this case it should be removed from the timeline
            try:
                self._trim_block_start(move_block, block.len)
                self.timeline[new_idx] = move_block
            except TimelineBlockError as e:
                pass

        # we cant edit a dictionary we are currently looping over but we may need to change the key or delete an entry
        # so for now, using this list to queue up orders to loop over after
        dict_orders = []
        # this loop checks for any overlapping blocks and trims them accordingly
        # preference is given to the newly inserter block
        for (check_idx, check_block) in self.timeline.items():
            try:
                if idx < check_idx < idx + block.len():
                    # if the start index of a previous block falls within the span of the new block, change its start position
                    # we trim the starting point of the block and then queue a key change to the new location
                    self._trim_block_start(check_block, block.len() - (check_idx - idx))
                    dict_orders.append((check_idx, idx+block.len()))
                elif check_idx < idx and check_idx+check_block.len() > idx:
                    # if the start index is before the new block, but the end index overlaps, we change the end position
                    self._trim_block_end(check_block, check_block.len() + (check_idx - idx))
            except TimelineBlockError as e:
                # this is for the case that the start and end index of an old block fall within the new blocks span
                # in this case we must delete the old block
                dict_orders.append((check_idx, -1))

        # this just applies the needed changes to the dict now that the loop is over
        for prev_idx, new_idx in dict_orders:
            if new_idx == -1:
                del self.timeline[prev_idx]
            else:
                self.timeline[new_idx] = self.timeline.pop(prev_idx)

        # with any clashes now dealt with, we can assign the new block its position on the timeline
        self.timeline[idx] = block

    def get_time_series(self):
        # length of array will encompass the whole timeline
        # this can be found by the key of the final entry + its length
        last_key = max(self.timeline)
        ts = np.zeros(last_key + self.timeline[last_key].len())

        for (pos, block) in self.timeline.items():
            add_ts = block.get_time_series()
            print(f"adding at pos: {librosa.samples_to_time(pos)//60:02.0f}:{librosa.samples_to_time(pos)%60:.2f} with length: {len(add_ts)}")
            ts[pos:pos+len(add_ts)] = add_ts
        return ts

    def _trim_block_start(self, block, trim_size):
        new_start = block.start_idx + trim_size

        try:
            block.set_start_idx(new_start)
        except TimelineBlockError as e:
            raise TimelineBlockError(e)

    def _trim_block_end(self, block, trim_size):
        new_end = block.end_idx - trim_size

        try:
            block.set_end_idx(new_end)
        except TimelineBlockError as e:
            raise TimelineBlockError(e)

class TimelineBlockError(Exception):
    pass

class TimelineBlock:
    """
    This class is used as a container for holding an audio source. This could be a midi sequence, an audio stream, or even a function.
    
    The contents are played back using an index, which allows portions of the block to be placed on the timeline with the ability to alter the trim locations. 

    Class Variables
    ---------------
    start_idx : number > 0
      * starting index of the block

    end_idx : number > 0
      * ending index of the block

    master_volume : float [0, 1]

    master_pan : float [-1, 1]
      * left (negative)/right (positive) pan of the block
    
    Class Methods
    -------------

    """
    def __init__(self, audio_handler) -> None:
        self.start_idx = 0
        self.end_idx = audio_handler.len()
        self.master_volume = 1
        self.master_pan = 0
        self.set_audio_handler(audio_handler)

    # TODO: verify handler is valid
    # set start/end pos depending on the handlers length and previous start/end
    def set_audio_handler(self, handler):
        self.audio_handler = handler

    def get_time_series(self):
        return self.audio_handler.get_time_series(self.start_idx, self.end_idx)

    def len(self):
        return (self.end_idx - self.start_idx)
    
    def set_start_idx(self, idx):
        if idx < 0:
            raise TimelineBlockError(f"Index {idx} is invalid: must be a positive integer")
        elif idx < self.audio_handler.len():
            self.start_idx = idx
        else:
            raise TimelineBlockError(f"Index {idx} is greater than this blocks audio source length of {self.audio_handler.len()}")
        
    def set_end_idx(self, idx):
        if idx < 0:
            raise TimelineBlockError(f"Index {idx} is invalid: must be a positive integer")
        elif idx < self.audio_handler.len():
            self.end_idx = idx
        else:
            raise TimelineBlockError(f"Index {idx} is greater than this blocks audio source length of {self.audio_handler.len()}")

import librosa
import numpy as np

class AudioStreamHandler:
    """
    Handler for holding streamed audio.
    Can be initiated using either a filename to load, or an already loaded sequence

    Class Variables
    ---------------

    Class Methods
    -------------
    """
    
    # Constants
    DEFAULT_SR = 22050
    
    def __init__(
            self, 
            filename: str = None, 
            ts: np.ndarray = None, 
            sr: int = None
            ) -> None:
        
        if filename:
            self.load_file(filename)
        elif ts:
            self.load_stream(ts, sr)

    def len(self):
        return len(self.ts)

    def load_file(self, filename) -> None:
        self.ts, self.sr = librosa.load(filename)

    def load_stream(self, ts, sr) -> None:
        self.ts = ts
        if sr:
            self.sr = sr
        else:
            self.sr = self.DEFAULT_SR

    def get_time_series(self, start_idx, end_idx):
        return self.ts[start_idx:end_idx]

# just testing functionality
if __name__ == "__main__":
    from timeline import *
    tl = Timeline()        
    stream1 = AudioStreamHandler("media/goingmad.mp3")
    stream2 = AudioStreamHandler("media/sued.flac")    
    stream3 = AudioStreamHandler("media/edcast.mp3")     
    stream4 = AudioStreamHandler("media/thisisacall.mp3")     
    stream5 = AudioStreamHandler("media/beat.wav")     
    tl.new_block(stream1)                     
    tl.new_block(stream2)                     
    tl.new_block(stream3) 
    tl.new_block(stream4) 
    tl.new_block(stream5)

    pos_list = []
    pos = 10_000
    pos_list.append(pos)

    pos += 5_000_000
    pos_list.append(pos)

    pos += 2_000_000
    pos_list.append(pos)

    pos += 2_000_000
    pos_list.append(pos)

    pos += 5_000_000
    pos_list.append(pos)

    tl.insert_block(pos_list[0], tl.block_list[0])
    tl.insert_block(pos_list[1], tl.block_list[1])
    tl.insert_block(pos_list[3], tl.block_list[3])
    tl.insert_block(pos_list[2], tl.block_list[2])
    tl.insert_block(pos_list[4], tl.block_list[4])
    for items in tl.timeline.items():
        print(f"{items[0]} -- {items[0] + items[1].len()} [{items[1].len()}]")
        print(f"{items[1].start_idx} -- {items[1].end_idx}\n")

    ts = tl.get_time_series()
    
    import soundfile as sf
    sf.write("out/timeline.wav", ts, 22050)
