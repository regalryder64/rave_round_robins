---

# rave_round_robins

* **`generate_snares`**: insert path to a sample and the `.ts` file you want to use, creates 20 samples with variance defined by the variance variable
* **`padscript`**: short script to pad the length of our audio files to 1.5 seconds to work with the training
* **`.ts` files**: exported models

---

## Order of operations:

1. **Get dataset**
2. **Pad files** with `padscript` if they are shorteer than 1.5s
3. **Preprocess**:
```bash
rave preprocess --input_path mediumsnare_wav --output_path output --num_signal 65536

```


4. **Train**:
```bash
rave train --config v2_small --db_path output_large --name large_run --val_every 2000 --channels 1 --save_every 10000 --batch 2 --n_signal 65536

```


5. **Tensorboard**:
```bash
tensorboard --logdir runs

```


> Use the tensorboard to monitor training, including listening to the audio in the audio tab


6. **Resume training**:
```bash
rave train --config v2_small --db_path output_large --name large_run --val_every 2000 --channels 1 --save_every 10000 --batch 2 --n_signal 65536 --ckpt runs/large_run_blahblah/version_blahblah/checkpoints/epoch_blah.ckpt

```


7. **Force to phase 2**: Stop training, access config file (`nano runs/large_run_YOUR_HASH/config.gin`), change `PHASE_1_DURATION` to 1 (or whatever step you want to switch to phase 2 on), then resume training as normal
8. **Export**:
```bash
rave export --run runs/large_run_blahblah --ckpt runs/large_run_blahblah/version_blahblah/checkpoints/epoch_blahblah.ckpt

```


9. **Generate**: use `generate_snares`

---

Would you like me to go through and fix the minor typos (like "shorteer") for you, or are you good to copy and paste this as is?
