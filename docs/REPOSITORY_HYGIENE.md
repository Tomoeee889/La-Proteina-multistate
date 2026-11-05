# Keeping generated data out of Git

## What was cleaned

Generated baseline MSA alignments, FASTA outputs, plots and job output files were removed from the current Git tree. Source code, scripts, configuration files, documentation, papers, assets used as documentation, and the ProteinMPNN gitlink were retained. Existing Python/shell files in temporary or notebook-checkpoint directories were retained because they may contain unique code.

This is a normal cleanup commit. Removed data remain in older commits; repository history and historical clone size are not reduced.

## Before updating an existing server checkout

A push changes GitHub; it does not contact or modify another checkout. However, pulling this cleanup commit can delete the previously tracked files from that checkout. Back up required experiment directories outside the repository BEFORE pulling, even when they will be ignored afterwards. Do not assume that .gitignore protects files already tracked in older commits.

For a Bash/Linux server, first inspect the existing checkout with:

```bash
git status --short
```

Copy the experiment directories you need to a separate backup location outside this checkout using your normal backup procedure, including local files not recorded by Git. Check that the backup is complete and readable. Also back up any modified tracked files. Then review the incoming change and update with a fast-forward-only pull when the working tree permits:

```bash
git fetch origin
git diff --stat HEAD..origin/main
git pull --ff-only
```

If Git reports local modifications or diverging branches, stop and resolve them without discarding data. After the update, restore only the generated data you need from the backup. Their paths are now ignored. Keep the backup until the restored results have been verified. This cleanup does not run any command on a remote server.

## Before every commit

Prefer adding named source files instead of blindly staging the whole checkout:

```bash
git add path/to/script.py path/to/config.yaml
git diff --cached --stat
git diff --cached --name-status
git diff --cached
```

Check ignore rules and tracked files with:

```bash
git status --short --ignored
git check-ignore -v --no-index path/to/result
git ls-files -- path/to/result
```

.gitignore affects untracked files. For a NEW accidentally tracked result, first add its precise output path to .gitignore, then use:

```bash
git rm --cached -- path/to/result
git add .gitignore
git diff --cached --stat
```

--cached leaves that file in the checkout where the command is run. Committing and pushing its removal can still delete the tracked copy when another machine pulls, so coordinate backups there first. Do not use git clean -fdx for routine cleanup: it can delete ignored results and other local data.

## Rule design

Ignore generated output directories, not every .json, .yaml, .txt, .pdb or .fasta file. Those extensions can also contain configuration, source inputs and examples. New run locations need corresponding narrow rules. Avoid git add -f for results. The ProteinMPNN directory is a gitlink, so its own repository has separate tracking and ignore rules.
