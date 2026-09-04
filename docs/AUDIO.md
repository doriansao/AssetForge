# Audio engine

Generates music and sound effects locally with Meta's AudioCraft, then
post-processes them into assets a game can actually use. See
[MODELS.md](MODELS.md) for model choice.

## Running it

The audio engine has **its own virtualenv** and must be run with that
interpreter. AudioCraft pins an older torch than the image engine uses, so they
cannot share an environment.

```bash
./engines/audio/venv/Scripts/python.exe engines/audio/scripts/audio_presets.py --list
./engines/audio/venv/Scripts/python.exe engines/audio/scripts/audio_presets.py \
    --preset sword_hit --output out/audio/sword.wav
```

There is no server to start. Models download on first use into `.cache/` at the
repository root, roughly 6GB per medium model.

## Presets

`scripts/audio_presets.py` covers what a 2D game needs: interface sounds,
pickups, movement, combat, world effects, and looping music for town, battle,
boss, menu, dungeon and victory.

```bash
<venv python> audio_presets.py --preset town_theme --output out/audio/town.wav
<venv python> audio_presets.py --preset footstep_stone --output out/audio/step.wav --count 4
```

`--count` generates variations from one prompt, which is what you want for
footsteps and impacts: shipping four and picking randomly at runtime is how games
avoid the machine-gun effect of one repeated sample.

`--extra` appends to the preset prompt without replacing it.

## Prompting

Both models respond to **material, impact, decay and space** far more than to the
name of a thing. This is the single highest-leverage thing to get right.

Weak: `sword hit`

Strong: `a steel blade striking a wooden shield, sharp crack with a short woody
decay, close and dry`

Name what is hitting what, what the sound does over time, and what room it is in.
"Close and dry" suppresses reverb, which you generally want, because reverb is
better added by the game engine per-space than baked into the sample.

Keep durations short. A UI click wants a fraction of a second; generating four
seconds and trimming wastes time and invites the model to add material you did
not ask for.

## Post-processing

`scripts/audio_processor.py`, usable standalone or as a library. Raw AudioCraft
output is not shippable: it starts with silence, ends mid-phrase, sits at an
arbitrary level, and clicks when a game starts or stops it.

| Step | Why |
|---|---|
| Trim | Leading silence on an effect is input lag the player feels |
| Resample to 44.1kHz | What engines expect; both models output lower |
| Loop (music only) | Equal-power cross-fade of tail over head |
| Fade | A waveform cut mid-cycle clicks on playback |
| Normalise to -1 dBFS | One mixer volume works for a whole folder |

```bash
<venv python> audio_processor.py -i raw.wav -o clean.wav --kind music
```

**Effects must not loop.** A loop cross-fades the tail over the attack, which
destroys the transient that makes an impact read as an impact. The default is
loop on for music, off for effects; `--loop` and `--no-loop` override.

The loop step is the same trick the image engine uses to make a texture tile:
blend the end into the beginning so the wrap is continuous. The *measurement*
does not transfer. The processor reports the wrap as a percentile of ordinary
sample-to-sample steps, not as a ratio against their mean, because audio step
distributions are heavily skewed: on a real track the mean step was 0.015 while
the maximum was 0.224, so a wrap 3.4x the mean is unremarkable. Anything under
roughly p99 is inaudible; values near p100 mean the cross-fade did not happen.

Equal-power curves are used rather than linear ones, so the blend does not dip in
perceived loudness halfway through.

## Output format

16-bit PCM WAV at 44.1kHz. That is the right default for effects, where decode
latency matters and every engine loads WAV without a codec. For long music tracks
you may want to convert to OGG Vorbis afterwards to save space; the engine does
not do this for you because the right bitrate depends on your budget.

## Licence warning

AudioCraft's **code** is MIT, but its **weights are CC-BY-NC 4.0**, which is
non-commercial. If you are shipping a commercial game, treat MusicGen and
AudioGen output as prototyping material. Stable Audio Open has a more permissive
community licence and better fidelity; see [MODELS.md](MODELS.md).

## Adding a model

Add an entry to `engines/audio/scripts/models.py` with its repo id, kind, sample
rate and VRAM. If it is not an AudioCraft model it will also need a loader branch
in `generate_audio.py`, since the current one dispatches between `MusicGen` and
`AudioGen` only.
