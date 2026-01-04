# Claude Command: update_git

Analyze git diff, validate changes are commit-worthy, optionally update changelog/version, then commit and push.

## Behavior (based on prompt)

| Prompt | Action |
|--------|--------|
| omitted / empty | Commit changes only (no updates) |
| `changelog` | Update CHANGELOG.md → commit |
| `version` | Update version + CHANGELOG.md → commit |

## Steps

### 1. Git Diff Analysis
Run `git diff --no-color` to get the current changes and parse file statuses (A=Added, M=Modified, D=Deleted).

### 2. Validate Changes for Commit

Review all changed files and use judgment to determine if they are appropriate to commit:

- If changes appear to be temporary files, build artifacts, debug outputs, or unrelated files that shouldn't be committed, report this to the user and ask for confirmation before proceeding
- Consider context: Is this a legitimate change for this repository? Does it make sense to version these files?
- If unsure, err on the side of caution and alert the user

### 3. Update Based on Prompt

**If prompt is `changelog`:**
- Read CHANGELOG.md
- Find or create unreleased section (`## [Unreleased]`)
- Categorize changes by component (top-level directory)
- Add entries to proper sections (Added/Changed/Fixed/etc.)
- Write updated CHANGELOG.md

**If prompt is `version`:**
- Read current version from `pyproject.toml`
- Bump version (patch by default, e.g., `2.0.0` → `2.0.1`)
- Update version in `pyproject.toml`
- Create new version section in CHANGELOG.md with today's date
- Move unreleased entries under the new version header

**If prompt is empty/omitted:**
- Skip changelog and version updates

### 4. Commit & Push
1. Run `git add .` to stage all changes
2. Generate a commit message based on the diff
3. Commit with the generated message
4. Push to remote with `git push`

## Notes
- Version bump uses patch level by default (x.y.0 → x.y.1)
- Date format: YYYY-MM-DD
- Follows Keep a Changelog format
