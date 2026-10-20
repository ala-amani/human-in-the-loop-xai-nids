# Changelog

## Unreleased

- Moved the repository-hygiene term list out of `tests/test_repository_hygiene.py` into an untracked local file (`tests/.hygiene_terms.txt`) or the `HYGIENE_TERMS` environment variable, so the published snapshot no longer contains the terms it screens for; the test skips when no terms are configured.
- Created the independent, data-free detection project.
