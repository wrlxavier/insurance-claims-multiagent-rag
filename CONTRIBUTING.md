# Contributing

## Branching model

Work happens on topic branches named `<area>/<subject>`, for example
`data/add-susep-corpus` or `feature/7-m0-07-ci-and-doc`. `<area>` is a short
label for what the change touches (`feature`, `data`, `docs`, `fix`, ...);
`<subject>` is a brief, hyphenated description, optionally prefixed with the
issue number.

- Topic branches are created from `staging` and merged back into `staging`
  via pull request.
- `staging` is periodically promoted to `main` via pull request, once the
  changes on it are verified.
- `main` reflects the current released/reviewable state of the project.

## Commit messages

Commits follow `type(scope): subject`, e.g. `feat(ci): add CI workflow` or
`docs(compliance): add canonical scope statement`. Common types: `feat`,
`fix`, `docs`, `refactor`, `test`, `chore`.

## Before opening a pull request

Run the full quality gate locally:

```bash
make check
```

This runs lint, format check, type check and the unit test suite — the same
checks enforced by CI on every push and pull request (see
`.github/workflows/ci.yml`). Optionally, install the pre-commit hooks so
these checks run automatically before each commit (see `README.md` for
setup instructions).

Every pull request must use the template in
`.github/pull_request_template.md`, which checks that tests, docs and
evaluation impact were considered.

## Branch protection

Branch protection rules (requiring the CI check to pass and requiring PR
review before merging into `staging` and `main`) are configured manually in
the repository's GitHub settings (**Settings → Branches**). They are not
enforced by any file in this repository.
