## Muser
Live interface to JACK MIDI synthesizers with audio capture and manipulation of musical entities.

### Requirements
#### For core features
- Python 3.x
- NumPy, SciPy
- The [JACK Audio Connection Kit](http://jackaudio.org/) and its Python [bindings](http://jackclient-python.readthedocs.io) for MIDI and audio interfacing.
  - Depends on C Foreign Function Interface (CFFI) [for Python](https://pypi.python.org/pypi/cffi).
- [music21](http://web.mit.edu/music21/) for musical entity representation and musicological analysis.

#### For auxiliary features
##### muser.fft
- The [gpyfft](https://github.com/geggo/gpyfft) Python wrapper for AMD's [clFFT](http://clmathlibraries.github.io/clFFT/)
  - Depends on OpenCL and its Python bindings (pyopencl).
##### muser.vis
- matplotlib
- moviepy: for creation of animated plots
- peakutils: for detection of peaks in 1D data

### Modules
#### Core
##### muser.audio
Audio file I/O and audio data manipulation.

##### muser.live
Live MIDI synthesizer interface with audio capture. The interface is pseudo-real-time due to Python's garbage collection; if these features are called upon in an application with large or frequent memory operations, JACK audio buffer under/overruns may result.
- Classes to handle the CFFI MIDI and audio ringbuffers that JACK requires for real-time interfacing.
- Classes that extend `jack.Client` with simplified port registration, xrun logging, dismantling, synthesizer interfacing, and audio capture.
- An experimental synthesizer class.

##### muser.sequencer
Representation and manipulation of musical entities. Interface between `music21` objects and MIDI messages.
- Generation of random `music21` notes and chords.
- Generation of random velocity vectors (vector representations of groups of MIDI notes).
- Conversion of velocity vectors to lists of MIDI events.
- Synthesis of MIDI control events.
- Reading and conversion of MIDI files.
- Harmonic and rhythmic transformations.

##### muser.utils
Utility classes and functions.
- Decorators for conditional and logged execution of methods.
- Thread manager classes for instance methods, dumping files.
- Functional generation of batches.
- General calculations and data manipulations.
- Unit conversions.

#### Auxiliary
##### muser.fft
Fourier analysis of audio data. Leverages OpenCL for fast transforms of large audio datasets.

##### muser.vis
Visualization routines.

### Scripts
#### chord_batches.py
Capture synthesizer audio for each of a batch of random chords.
#### midi_play.py
Simple playback of MIDI files through synthesizer. Currently only accounts for note on and off events and tempo changes.
#### synth_tone.py
Basic test of tone synthesis with `muser.live.Synth`

### TODO
- Dependency versions in setup.py.
- Unit tests and test integration.
- Sphinx & ReadTheDocs.
