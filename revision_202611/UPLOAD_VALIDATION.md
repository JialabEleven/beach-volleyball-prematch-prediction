# Upload preparation validation

Prepared 12 November 2026 against the public repository snapshot e5f16edc64fdf813515ceeccde11101d340f8d1c.

On the authors' Linux server, the CLI help and six temporal invariant tests passed. The dedicated current S18 entry point reproduced both reference comparison intervals and all 5,000 bootstrap draws under each grouping definition within 1e-12. The current S19 bootstrap likewise reproduced all six comparison intervals and all draws within 1e-12. Three current entry-point scripts compiled successfully.

No full model retraining, clean-environment installation, new network raw-data download or all-analysis reproduction was performed for this upload preparation. Earlier controlled fits and v2 execution checks are documented in VALIDATION_REPORT.md and reference manifests. Supplementary S16/S18/S19 workbooks are copied from the previously checked standalone exports; no publication-PDF layout verification is implied.

The integrity checker excludes generated outputs, downloaded data, virtual environments and Python bytecode while verifying packaged source/reference files. File hashes are regenerated for this upload bundle because the README, current S18 entry point and table files were added or updated. Exact numerical seed settings remain in code and configuration records. Historical author response drafts were omitted from this public-code increment; technical audit records remain.
