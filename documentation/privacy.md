# Privacy and data ownership

Local Voice Pipeline processes recordings on the machine where it runs. The library has no cloud recognition service and does not need an account.

## What stays local

- Training recordings
- Wake-phrase and negative examples
- Recent diagnostic trigger captures
- Optional response audio
- Your command configuration

The integration you attach may communicate with another system. Review that adapter separately; the core recognizer does not make an external action by itself.

## Treat recordings as personal data

A voice recording can identify or reveal information about a person. Store the data folder privately, restrict access to the service user, and include recordings only in encrypted or otherwise protected backups.

Do not commit real household recordings, private response assets, automation tokens, entity IDs, or local hostnames to a public repository. Public tests should synthesize audio or use recordings with clear redistribution permission.

## Diagnostic retention

Wake-phrase mode stores a bounded queue of recent triggers for review. The current default keeps up to 20 trigger events. Dismiss unneeded events in the Studio, and periodically review the recordings folder as part of normal maintenance.

## Before sharing a bug report

Prefer logs, sanitized thresholds, and synthetic reproductions. If audio is essential, trim it to the minimum needed, remove unrelated speech, and share it only through a channel appropriate for personal data.
