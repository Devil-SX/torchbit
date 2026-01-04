# Claude Command: update_changelog

Analyze git diff and add changes to the newest unreleased version in CHANGELOG.md.

## Steps

1. Run `git diff --no-color` to get the current changes.

2. Parse the diff output to identify changed files with their status (A = Added, M = Modified, D = Deleted).

3. Read `CHANGELOG.md` to find the unreleased section:
   - Look for `## [Unreleased]` or similar markers
   - If no unreleased section exists, create one at the top

4. Categorize each change based on the file path:
   - Analyze the directory structure to determine the component/module name
   - Group changes by their general area (e.g., based on top-level directory or file naming patterns)

5. Format each change entry following the Keep a Changelog style:
   - Use appropriate section headers: Added, Changed, Deprecated, Removed, Fixed, Security
   - Format entries consistently with existing changelog entries

6. Update the CHANGELOG.md:
   - Add new entries to the unreleased section under proper headers
   - Use date format `YYYY-MM-DD` when creating a new version header

7. Write the updated changelog and show the user a summary of changes made.
