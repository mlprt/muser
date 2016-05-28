import sys
import time
import rtmidi
import music21
import binascii
import struct

NOTE_ON = 0x90
NOTE_OFF = 0x80
""" MIDI parameters. """




def init_midi_out():
    midi_out = rtmidi.MidiOut()
    midi_in = rtmidi.MidiIn()

    ports_out = midi_out.get_ports()
    ports_in = midi_in.get_ports()

    if ports_out:
        midi_out.open_port(0)
    else:
        midi_out.open_virtual_port("Virtual Port Out 0")

    if ports_in:
        midi_in.open_port(0)
    else:
        midi_in.open_virtual_port("Virtual Port In 0")

    return midi_out, midi_in


def get_send_events(midi_out):

    def send_events(events, tempo=90, loop=False):
        """ S """
        while True:
            for event, pause in events:
                midi_out.send_message(event)
                time.sleep(pause)
            if loop > 1:
                loop -= 1
            if not loop:
                break

    return send_events


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


def to_midi_notes(music21_notes, switch=NOTE_ON):
    """ Convert music21 notes to MIDI note tuples. """
    try:
        return (switch, music21_notes.pitch.midi, music21_notes.volume.velocity)
    except AttributeError:
        return tuple(to_midi_notes(note, switch) for note in notes)


if __name__ == '__main__':

    tempo = 90/60.
    melody = music21.converter.parse("tinynotation: C8 D# G c G D#")
    notes = list(melody.flat.getElementsByClass(music21.note.Note))
    durations = []
    for note in notes:
        note.duration.quarterLength = 0.1
        note.volume.velocityScalar = 0.7
        durations.append(tempo * note.duration.quarterLength)
    midi_notes_on = to_midi_notes(notes)
    midi_notes_off = to_midi_notes(notes, NOTE_OFF)
    midi_notes = []
    for i, note in enumerate(midi_notes_on):
        midi_notes.append((note, durations[i]))
        midi_notes.append((midi_notes_off[i], 0))

    try:
        midi_out, midi_in = init_midi_out()
        send_events = get_send_events(midi_out)
        while True:
            send_events(midi_notes)

    except (KeyboardInterrupt, SystemExit):
        try:
            del midi_out
        except NameError:
            pass
        raise
