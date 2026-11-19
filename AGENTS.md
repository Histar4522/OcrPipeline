# AGENTS.md

## Workflow & Git Hygiene
- **Scope:** Keep changes strictly scoped to the requested task; avoid unrequested refactors.
- **Atomic Commits:** Commit locally after completing each discrete, verified task using concise, imperative messages (e.g., `fix: resolve null check in parser`).
- **No Remote Pushes:** Never run `git push`, force-push, or modify remote branches unless explicitly instructed.
- **Clean State:** Leave the working tree clean before concluding a session; do not commit temporary files, logs, or secrets.

## Code Quality & Verification
- **Verify Before Committing:** Run relevant tests, linters, and type/build checks before every commit. Fix any regressions introduced by your changes.
- **Reuse First:** Prefer existing project utilities and standard ecosystem libraries over writing custom boilerplate.
- **Minimal Comments:** Write self-documenting code; add brief docstrings/comments only for non-obvious invariants or public APIs.

## Safety & Execution Boundaries
- **Destructive Actions:** Ask for confirmation before deleting files, dropping data, altering schemas, or modifying environment/CI configurations.
- **Dependencies:** Do not add or upgrade third-party dependencies without explicit approval.
- **Stuck Protocol:** If blocked by ambiguity or repeated test failures after 2 attempts, stop and ask for clarification rather than guessing.