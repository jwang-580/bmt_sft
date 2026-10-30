# Data and model availability

The fine-tuned model is available at [JWang580/BMT-GRAFT-12B](https://huggingface.co/JWang580/BMT-GRAFT-12B). The training dataset and admission split are planned for release through PhysioNet; the accession and access conditions are forthcoming.

The repository contains code, question metadata, a study-design figure, and the per-question answer-match indicators used for Figure S2. Clinical notes, training conversations, evidence quotations, rationales, raw responses, and note caches are excluded. MIMIC-derived clinical inputs require the applicable source-data access and reuse authorization. Institutional clinical records are not distributed.

Extracted answers and aggregate results may be shared after review. The [answer-only exporter](docs/publication.md#export-answers) removes narrative fields and replaces record identifiers. Review dates and free-text answer values for the intended release before adding exports under `public_results/`.

For source packaging and the handling of clinical files in earlier Git commits, see [source releases](docs/publication.md#build-a-source-archive).
