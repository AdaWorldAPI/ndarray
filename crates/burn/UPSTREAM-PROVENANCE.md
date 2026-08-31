# Upstream provenance — where the reference tree went

`crates/burn/upstream` was a git SUBMODULE pointing at
`https://github.com/AdaWorldAPI/burn.git` (last gitlink
`9b2b67127b0fbb5387021faf540b7b12b9c4e943` = that repo's merge of PR #4;
upstream lineage: tracel-ai/burn, `v0.21.0-pre.2`). It was pure reference
material: `crates/burn/src` reads nothing from it, and `crates/burn` is
workspace-EXCLUDED besides.

It is gone because a reference-only submodule taxes EVERY consumer of this
repo as a git dependency: cargo clones git deps with submodules
unconditionally, so `medcare-rs`'s Railway build fetched the full burn fork
just to compile `chacha20` out of this repo — and on 2026-08-31 that fetch
died on GitHub's unauthenticated shared-IP rate limiting
("could not read Username", after three spurious-network retries), taking
the deploy down. A gitlink is also a commit-pin on an internal sibling —
the consumer pin law's spirit, one layer down.

Need the reference tree? Clone the sibling directly:
`git clone https://github.com/AdaWorldAPI/burn.git` — the fork repo is the
address; this file is the provenance record (tags over pins, per the
internal-pin prohibition).
