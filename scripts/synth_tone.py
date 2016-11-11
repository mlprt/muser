"""Basic test of tone synthesis with `muser.live.Synth`"""

import muser.live as live
import math
import time

synth = live.Synth(channels=2)

def synth_on(synth, duration, pause=0):
    """Activate synth's tone generation for a time, then pause."""
    synth.toggle()
    time.sleep(duration)
    synth.toggle()
    time.sleep(pause)

def sine_tone(amp, freq, phase=0):
    """Returns a sinusoidal function of time."""
    def tone(t):
        return amp * math.sin(2 * math.pi * freq * t + phase)
    return tone

synth.activate()
synth.connect(synth.outports[0], 'system:playback_1')
synth.connect(synth.outports[1], 'system:playback_2')

# add a 440 Hz tone and play twice for 1s, with a 0.5 pause between 
synth.add_synth_function(sine_tone(0.75, 440))
for i in range(2):
    synth_on(synth, duration=1, pause=0.5)

# repeat, adding a tone an octave below
synth.add_synth_function(sine_tone(0.75, 220))
for i in range(2):
    synth_on(synth, duration=1, pause=0.5)

synth.deactivate()
synth.outports.clear()
synth.close()
