---
name: commit
description: Analyze git diff, update CHANGELOG.md, then commit and push changes
argument-hint: [version]
---

# Claude Command: commit

Analyze git diff, update CHANGELOG.md, then commit and push changes.

## Behavior (based on prompt)

| Prompt | Action |
|--------|--------|
| omitted / empty | Update CHANGELOG.md → commit → push |
| `version` | Update version + CHANGELOG.md → commit → push |

## Steps

### 1. Git Diff Analysis

Run `git diff --no-color` and `git diff --cached --no-color` to get all changes and parse file statuses (A=Added, M=Modified, D=Deleted).

### 2. Validate Changes for Commit

Review all changed files and use judgment to determine if they are appropriate to commit:

- If changes appear to be temporary files, build artifacts, debug outputs, or unrelated files that shouldn't be committed, report this to the user and ask for confirmation before proceeding
- Consider context: Is this a legitimate change for this repository? Does it make sense to version these files?
- If unsure, err on the side of caution and alert the user

### 3. Update CHANGELOG.md

1. Read `CHANGELOG.md` to find the unreleased section:
   - Look for `## [Unreleased]` or similar markers
   - If no unreleased section exists, create one at the top

2. Categorize each change based on the file path:
   - Analyze the directory structure to determine the component/module name
   - Group changes by their general area (e.g., based on top-level directory or file naming patterns)

3. Format each change entry following the Keep a Changelog style:
   - Use appropriate section headers: Added, Changed, Deprecated, Removed, Fixed, Security
   - Format entries consistently with existing changelog entries

4. Update the CHANGELOG.md:
   - Add new entries to the unreleased section under proper headers
   - Use date format `YYYY-MM-DD` when creating a new version header

### 4. Version Update (if prompt is `version`)

1. Read current version from `pyproject.toml`
2. Bump version (patch by default, e.g., `2.0.0` → `2.0.1`)
3. Update version in `pyproject.toml`
4. Create new version section in CHANGELOG.md with today's date
5. Move unreleased entries under the new version header

### 5. Commit & Push

1. Run `git add .` to stage all changes
2. Generate a commit message based on the diff
3. Commit with the generated message
4. Push to remote with `git push`

## Notes

- Version bump uses patch level by default (x.y.0 → x.y.1)
- Date format: YYYY-MM-DD
- Follows Keep a Changelog format
