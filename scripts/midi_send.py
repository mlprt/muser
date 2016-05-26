import sys
import time
import jack
import music21
import binascii
import struct

client = jack.Client("TestClient")
inport = client.midi_inports.register("input")
outport = client.midi_outports.register("output")

# manual connection to pianoteq
input()

stream0 = music21.stream.Stream()

notes = ['c3', 'd#3', 'g3', 'c4', 'g3', 'd#3']

for n in notes:
    note = music21.note.Note(n)
    stream0.append(note)

blocksize = client.blocksize    # samples
sample_rate = client.samplerate # samples/s
blocktime = blocksize/sample_rate
print(blocktime)


def report_midi_event(event, last_frame_time=0, out=sys.stdout):
    """ Print details of a midi event. """
    offset, indata = event
    #print(struct.unpack(str(len(indata))+'B\n', indata))
    try:
        status, pitch, vel = struct.unpack('3B', indata)
    except struct.error:

        return
    rprt = "{0} + {1}:\t0x{2}\n".format(last_frame_time,offset,
                                     binascii.hexlify(indata).decode())
    #rprt += "indata: {0}\n".format(indata)
    rprt += "status: {0},\tpitch: {1},\tvel.: {2}\n".format(status, pitch, vel)
    #rprt += "repacked: {0}".format(struct.pack('3B', status, pitch, vel))
    rprt += "\n"
    out.write(rprt)

def map_midi_pitches(pitches):
    """ Convert midi pitches to music21 pitch objects. """
    pass


@client.set_process_callback
def process(frames):
    outport.clear_buffer()
    for event in inport.incoming_midi_events():
        report_midi_event(event, client.last_frame_time)
        outport.write_midi_event(*event)

def play_stream(stream, tempo=90, loop=False):
    while True:
        for note in stream:
            outport.clear_buffer()
            # TODO: Figure out timing... need to avoid xruns?
            # TODO: Convert music21 notes to midi events
            outport.write_midi_event(1, (144,38,100))
        if not loop:
            break

with client:
    print("#" * 80)
    print("press Return to quit")
    print("#" * 80)
    input()
