---

## rave_round_robins

**`app`**: launches a streamlink gui which will let you generate round robins based on whatever sample you upload
**`padscript`**: short script to pad the length of our audio files to 1.5 seconds to work with the training
**`.ts` files**: exported models

---

## Order of operations:

1. **Get dataset**
2. **Pad files** with `padscript` if they are shorter than 1.5s
> 2.5. If training many files, concatenate them into longer series of files.
3. **Preprocess**:
```bash
rave preprocess --input_path input_dataset --output_path preprocessed_dataset --num_signal 65536

```
> Use the --num_signal argument if your samples are short (around 1.5s). Otherwise, leave blank.

4. **Train**:
```bash
rave train --config v2_small --db_path preprocessed_data --name large_run --val_every 2000 --channels 1 --save_every 10000 --batch 2 --n_signal 65536

```

5. **Tensorboard**:
```bash
tensorboard --logdir runs

```


> Use the tensorboard to monitor training, including listening to the audio in the audio tab


6. **Resume training**:
```bash
rave train --config v2_small --db_path preprocessed_dataset --name large_run --val_every 2000 --channels 1 --save_every 10000 --batch 2 --n_signal 65536 --ckpt runs/large_run_blahblah/version_blahblah/checkpoints/epoch_blah.ckpt

```


7. **Force to phase 2**: Stop training, access config file (`nano runs/large_run_blahblah/config.gin`), change `PHASE_1_DURATION` to 1 (or whatever step you want to switch to phase 2 on), then resume training as normal

8. **Export**:
```bash
rave export --run runs/large_run_blahblah/version_blahblah

```


9. **Generate**: use the streamlink app, making sure your filepaths are correct

---
