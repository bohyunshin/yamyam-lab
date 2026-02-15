---
name: pr-reviewer
description: Review GitHub pull requests with code analysis, verification, and feedback. Use when the user asks to review a PR, provides a PR URL, or invokes /pr-reviewer. Analyzes code changes, runs project verification skills, posts line-level review comments, and approves or requests changes.
---

# PR Reviewer

## Workflow

### Step 1: Get Pull Request

If a PR URL or number is provided as an argument, use it directly. Otherwise, ask the user:

```
AskUserQuestion: "Which pull request should I review?"
```

Extract the PR number. Fetch PR metadata and diff:

```bash
gh pr view <number> --json number,title,body,author,baseRefName,headRefName,files,additions,deletions
gh pr diff <number>
```

Display a summary:

```markdown
## PR Review: #<number> — <title>

- **Author:** <author>
- **Branch:** <head> → <base>
- **Changes:** +<additions> / -<deletions> across <file_count> files
```

### Step 2: Analyze Code Changes

For each changed file in the diff:

1. Read the full file to understand context beyond the diff
2. Check for:
   - **Bugs** — Logic errors, off-by-one, null/None handling, race conditions
   - **Security** — Injection, hardcoded secrets, unsafe deserialization
   - **Style** — Naming conventions, code consistency with project patterns
   - **Design** — Unnecessary complexity, missing abstractions, duplicated logic
   - **Types** — Missing or incorrect type hints (Python)
   - **Tests** — Changed logic without corresponding test updates
   - **Config** — Missing or inconsistent configuration entries

Record each issue found with: file path, line number, severity (critical/warning/suggestion), description, and suggested fix.

### Step 3: Run Verification Skills

Checkout the PR branch locally and run `/verify-implementation`:

```bash
gh pr checkout <number>
```

Then invoke the `verify-implementation` skill. Record any issues found by the verification.

### Step 4: Summarize Analysis

Compile all findings into a review summary:

```markdown
## Review Summary

### Code Analysis

| # | File | Line | Severity | Issue | Suggestion |
|---|------|------|----------|-------|------------|
| 1 | `path/to/file.py` | 42 | critical | Description | Fix suggestion |
| 2 | `path/to/file.py` | 78 | warning | Description | Fix suggestion |

### Verification Results

| Skill | Status | Issues |
|-------|--------|--------|
| verify-test-coverage | PASS / FAIL | N |
| verify-code-convention | PASS / FAIL | N |
| ... | ... | ... |

### Overall Assessment

<Summary of PR quality, key concerns, and positive aspects>
```

Present this summary to the user before posting to GitHub.

### Step 5: Post Review Comments

Use `AskUserQuestion` to confirm before posting:

```
"Ready to post review to GitHub. Proceed?"
- Post review (Recommended)
- Edit findings first
- Skip posting
```

Post line-level review comments using the GitHub CLI:

```bash
# Submit a pull request review with inline comments
gh api repos/{owner}/{repo}/pulls/<number>/reviews \
  --method POST \
  -f body="<overall review summary>" \
  -f event="<APPROVE or REQUEST_CHANGES>" \
  -f 'comments[0][path]=<file>' \
  -f 'comments[0][line]=<line>' \
  -f 'comments[0][body]=<comment with suggestion>'
```

Use GitHub suggestion blocks in comment bodies for actionable fixes:

````markdown
**[severity]** Description of the issue.

```suggestion
corrected code here
```
````

### Step 6: Approve or Request Changes

Determine the review decision:

- **APPROVE** — No critical issues found, code is well-written
- **REQUEST_CHANGES** — Critical issues or multiple warnings exist

The decision is set via the `event` field in Step 5's API call.

After posting, display confirmation:

```markdown
## Review Posted

- **Decision:** APPROVE / REQUEST_CHANGES
- **Comments:** N inline comments posted
- **PR:** <PR URL>
```

## Exceptions

1. **Draft PRs** — Review normally but note draft status in summary
2. **Merge commits** — Skip merge commit diffs, focus on authored changes only
3. **Generated files** — Skip lock files, migration files, and build outputs
4. **Large PRs (>30 files)** — Focus on source code files, deprioritize config/docs unless relevant
