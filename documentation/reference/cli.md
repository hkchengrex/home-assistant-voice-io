# Command line

Global options come before the subcommand:

```text
voice-io --config PATH --recordings PATH COMMAND [OPTIONS]
```

| Command | What it does |
| --- | --- |
| `devices` | Lists available audio input devices. |
| `record NAME` | Records template WAV files for one command or special set. |
| `listen` | Captures and classifies one utterance. |
| `match WAV` | Classifies an existing WAV file. |
| `templates` | Shows the number of recordings for each command. |
| `calibrate` | Measures same-command and other-command distance separation. |
| `clone-response` | Generates one cloned response with `--text`, or a batch with `--text-file`. |
| `studio` | Starts the local Training Studio at port 8765 by default. |
| `run` | Runs continuous wake-phrase or direct listening. |

Use the built-in help for the exact options in your installed version:

```shell
voice-io --help
voice-io record --help
voice-io clone-response --help
voice-io run --help
```

## Special recording sets

- `_start_phrase` contains examples of the configured wake phrase.
- `_not_command` contains varied speech that should be rejected.

These names are configured by the library and are not external automation actions.

## Exit behavior

`listen` and `match` exit with status 0 for an accepted command and status 2 for a safe rejection. Configuration and missing-library problems stop with an explanatory error.
