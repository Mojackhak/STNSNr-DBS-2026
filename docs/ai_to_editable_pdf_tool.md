# AI to Editable PDF Tool

`tools/export_ai_to_layered_pdf.jsx` is an Adobe Illustrator ExtendScript tool for exporting Illustrator `.ai` files to editable layered PDFs. It is intended for high-resolution manuscript figure delivery where Illustrator editability and Acrobat layer metadata must be preserved.

## Requirements

- Adobe Illustrator 2026 installed at `/Applications/Adobe Illustrator 2026/Adobe Illustrator.app`
- macOS `osascript`
- Absolute input and output paths
- A JSON config file supplied at runtime through the global `AI_TO_PDF_CONFIG` variable

## Workflow

1. Build an explicit source-file list and choose the output directory. Check relevant worktree paths only when writing configs, outputs, or tool files inside the MyLFP repository, and follow its repository instructions; an export outside the repository does not require a Git status check.
2. Move any target PDFs being replaced to Trash, then create the config. The JSX already validates the config, source paths, and target conflicts; do not duplicate those checks outside the script.
3. Run the JSX and perform the default completion check below.

## Config

The JSX does not include any default input files. Callers must create a config JSON:

```json
{
  "sourceFiles": [
    "/absolute/path/Figure_01.ai",
    "/absolute/path/Supplementary_Figure_01.ai"
  ],
  "outputDir": "/absolute/output/folder",
  "logFile": "/tmp/ai_to_pdf_export.log",
  "overwriteExisting": false
}
```

Fields:

- `sourceFiles`: required array of absolute `.ai` file paths.
- `outputDir`: required absolute directory for same-basename `.pdf` outputs.
- `logFile`: optional absolute log path. Defaults to `/tmp/ai_to_pdf_export.log`.
- `overwriteExisting`: optional boolean. Defaults to `false`; when false, the script fails if a target PDF exists.

If a target PDF should be replaced, move the old PDF to Trash before running the script. Do not permanently delete untracked files that cannot be recovered through Git.

## Export Settings

The script saves each file with:

- Acrobat 6 / PDF 1.5 compatibility
- Illustrator editability preserved
- Acrobat layers preserved from top-level Illustrator layers
- Color, grayscale, and monochrome image downsampling set to 600 dpi
- Downsampling threshold set to 900 dpi
- Bicubic downsampling
- Maximum JPEG quality for color and grayscale images
- No document or permission password

## Run Command

```bash
osascript -e 'tell application id "com.adobe.illustrator" to do javascript "var AI_TO_PDF_CONFIG=\"/tmp/ai_to_pdf_config.json\"; $.evalFile(\"${MYLFP_REPO_ROOT}/tools/export_ai_to_layered_pdf.jsx\");"'
```

If the export fails, an expected output is missing or empty, or the AppleEvent times out, inspect the configured log and expected output paths before considering a retry; Illustrator may still be working or may have completed the export. This status check does not enable deep validation.

## Default Completion Check

After a successful export, check once that each expected PDF exists and is non-empty. Use the output paths derived from `sourceFiles`, not the total file count in the output directory. Report the completed count and output location.

Do not inspect normal logs, check PDF structure, page count, version, or layer metadata, render previews, or reopen PDFs in Illustrator by default. Do not claim visual or editing behavior was verified when those checks were not requested and performed.

## Deep Validation (Explicit User Request Only)

Run deep validation only when the user explicitly requests it or a specific check. A request to export editable PDFs, publication use, an export error, or a timeout does not activate it. For a specific request, perform only the requested checks on the requested outputs.

For a full deep-validation request, check every expected output:

```bash
qpdf --check /absolute/output/Figure_01.pdf
qpdf --show-npages /absolute/output/Figure_01.pdf
file /absolute/output/Figure_01.pdf
strings /absolute/output/Figure_01.pdf | rg -m 1 '/OCProperties'
```

Expected results:

- one PDF per source file
- one page per PDF unless the source intentionally has multiple artboards/pages
- PDF 1.5 for Acrobat 6 compatibility
- `/OCProperties` present for Acrobat layer metadata

For full deep validation, also render representative pages to PNG for visual review and reopen at least one representative output in Illustrator to check editability. A blank PDF is acceptable only when the source `.ai` also renders blank.
