rave_round_robins

    generate_snares: insert path to a sample and the .ts file you want to use, creates 20 samples with variance defined by the variance variable

    padscript: short script to pad the length of our audio files to 1.5 seconds to work with the training

    .ts files: exported models

Order of operations:

    Get dataset

    Pad files with padscript if they are shorteer than 1.5s

    Preprocess:
    Bash

    rave preprocess --input_path mediumsnare_wav --output_path output --num_signal 65536

    Train:
    Bash

    rave train --config v2_small --db_path output_large --name large_run --val_every 2000 --channels 1 --save_every 10000 --batch 2 --n_signal 65536

    Tensorboard:
    Bash

    tensorboard --logdir runs

        Use the tensorboard to monitor training, including listening to the audio in the audio tab

    Resume training:
    Bash

    rave train --config v2_small --db_path output_large --name large_run --val_every 2000 --channels 1 --save_every 10000 --batch 2 --n_signal 65536 --ckpt runs/large_run_blahblah/version_blahblah/checkpoints/epoch_blah.ckpt

    Force to phase 2: Stop training, access config file (nano runs/large_run_YOUR_HASH/config.gin), change PHASE_1_DURATION to 1 (or whatever step you want to switch to phase 2 on), then resume training as normal

    Export:
    Bash

    rave export --run runs/large_run_blahblah --ckpt runs/large_run_blahblah/version_blahblah/checkpoints/epoch_blahblah.ckpt

    Generate: use generate_snares
