# Packaging notes

- Do not commit `bin/`, `build/`, `cache/`, `output*/`, or `plots/`.
- Runtime caches must be mounted or pointed to from outside the packaged tree.
- The package ships source, scripts, configs, docs, and validation reports only.
