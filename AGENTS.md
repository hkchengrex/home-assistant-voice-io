# Public repository guidance

- Keep this repository automation-neutral and cross-platform. Platform service
  integrations must be optional and guarded by runtime capability checks.
- Never add household recordings, spoken response assets, webhook IDs, entity
  IDs, private hostnames, credentials, or device-specific gain settings.
- Tests must synthesize audio or use explicitly redistributable fixtures.
- Preserve the `ha_voice` import namespace and public action/config contracts
  across compatible releases; document intentional breaking changes.
- Run the test suite and build both distributions before publishing a release.
