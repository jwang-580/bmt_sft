# Source releases and answer exports

## Build a source archive

The source release includes code, documentation, schemas, question definitions, synthetic tests, and reviewed answer-only tables. It also includes the R figure scripts, Figure 1, and the Figure S2 input restricted to question metadata and answer-match flags. Clinical notes, training conversations, quoted evidence, raw model responses, caches, and model weights are excluded.

The allowlist in `scripts/publication.py` determines which files are included. Each archive contains file hashes in `RELEASE_MANIFEST.json` and excludes Git history.

```bash
python scripts/publication.py check
python scripts/publication.py history
python scripts/publication.py package --output releases/BMT-GRAFT-12B-source.zip
```

`check` examines staged files in the Git index. `package` validates the current allowlisted files, including untracked files, and writes a ZIP archive. `history` reports excluded paths in earlier commits without reading their contents or changing history.

Clinical training data and the reserved admission list were tracked in earlier commits. Removing those files from the index does not remove historical copies. Resolve these historical files before making the existing Git history public, or distribute a source archive without history.

Automated checks detect disallowed paths, schemas, and credential patterns. Review changed source files and answer values before release; these checks do not deidentify clinical text.

## Export answers

```bash
python scripts/export_public_answers.py prompt_eval/results/private_predictions.json \
  --output public_results/registry_answers.json
```

The exporter retains answers, field identifiers, model configuration, and extraction statuses. It excludes evidence, rationales, raw responses, notes, note dates, source paths, and source identifiers. A fresh private salt replaces record identifiers for each export. Export paired model results together to preserve record matching; separate exports use different identifiers. The salt and identity mapping are not exported.

Validation rejects extra columns, nested answer objects, multiline or oversized strings, and unreplaced identifiers. Concise answer values, including dates, are allowed and require review before sharing. The output path must be new.

## Release checklist

- Run the tests, index check, and archive validation; review the files selected for release.
- Record the code revision, model and dataset versions, environment, and scoring policy.
- Add the PhysioNet accession after release and the manuscript identifier after publication.
- Confirm the applicable licenses for code, models, form content, and datasets.
