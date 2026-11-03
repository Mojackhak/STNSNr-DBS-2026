/*
Export Illustrator AI files to editable layered 600 dpi PDFs.

Usage from macOS:
osascript -e 'tell application id "com.adobe.illustrator" to do javascript "var AI_TO_PDF_CONFIG=\"/tmp/config.json\"; $.evalFile(\"/Users/mojackhu/Github/MyLFP/tools/export_ai_to_layered_pdf.jsx\");"'
*/

function fail(message) {
    throw new Error(message);
}

function isArray(value) {
    return value instanceof Array;
}

function isAbsolutePath(path) {
    return typeof path === "string" && path.length > 0 && path.charAt(0) === "/";
}

function hasAiExtension(path) {
    return /\.ai$/i.test(path);
}

function basenameWithoutExtension(file) {
    return file.name.replace(/\.[^.]+$/, "");
}

function defaultLogPath() {
    return "/tmp/ai_to_pdf_export.log";
}

function parseJson(text) {
    if (typeof JSON !== "undefined" && JSON.parse) {
        return JSON.parse(text);
    }
    return eval("(" + text + ")");
}

function readConfig() {
    if (typeof AI_TO_PDF_CONFIG === "undefined") {
        fail("AI_TO_PDF_CONFIG is required and must point to a JSON config file.");
    }
    if (!isAbsolutePath(AI_TO_PDF_CONFIG)) {
        fail("AI_TO_PDF_CONFIG must be an absolute path: " + AI_TO_PDF_CONFIG);
    }

    var configFile = new File(AI_TO_PDF_CONFIG);
    if (!configFile.exists) {
        fail("Config file does not exist: " + configFile.fsName);
    }
    if (!configFile.open("r")) {
        fail("Cannot open config file: " + configFile.fsName);
    }
    var text = configFile.read();
    configFile.close();

    var config = parseJson(text);
    if (!config || typeof config !== "object") {
        fail("Config must be a JSON object.");
    }
    if (!isArray(config.sourceFiles) || config.sourceFiles.length === 0) {
        fail("Config sourceFiles must be a non-empty array.");
    }
    if (!isAbsolutePath(config.outputDir)) {
        fail("Config outputDir must be an absolute path.");
    }
    if (config.logFile !== undefined && !isAbsolutePath(config.logFile)) {
        fail("Config logFile must be an absolute path when provided.");
    }
    if (config.overwriteExisting !== undefined && typeof config.overwriteExisting !== "boolean") {
        fail("Config overwriteExisting must be a boolean when provided.");
    }

    config.logFile = config.logFile || defaultLogPath();
    config.overwriteExisting = config.overwriteExisting === true;
    return config;
}

function openLog(logPath) {
    var logFile = new File(logPath);
    if (!logFile.parent.exists && !logFile.parent.create()) {
        fail("Cannot create log directory: " + logFile.parent.fsName);
    }
    if (!logFile.open("w")) {
        fail("Cannot open log file: " + logFile.fsName);
    }
    logFile.writeln("Starting AI to editable PDF export");
    logFile.close();
    return logFile;
}

function log(logFile, message) {
    logFile.open("a");
    logFile.writeln(new Date().toString() + " " + message);
    logFile.close();
}

function buildPdfOptions() {
    var options = new PDFSaveOptions();
    options.compatibility = PDFCompatibility.ACROBAT6;
    options.preserveEditability = true;
    options.acrobatLayers = true;
    options.generateThumbnails = true;
    options.compressArt = true;
    options.pDFXStandard = PDFXStandard.PDFXNONE;

    options.colorDownsampling = 600;
    options.colorDownsamplingImageThreshold = 900;
    options.colorDownsamplingMethod = DownsampleMethod.BICUBICDOWNSAMPLE;
    options.grayscaleDownsampling = 600;
    options.grayscaleDownsamplingImageThreshold = 900;
    options.grayscaleDownsamplingMethod = DownsampleMethod.BICUBICDOWNSAMPLE;
    options.monochromeDownsampling = 600;
    options.monochromeDownsamplingImageThreshold = 900;
    options.monochromeDownsamplingMethod = DownsampleMethod.BICUBICDOWNSAMPLE;

    options.colorCompression = CompressionQuality.AUTOMATICJPEGMAXIMUM;
    options.grayscaleCompression = CompressionQuality.AUTOMATICJPEGMAXIMUM;
    options.monochromeCompression = MonochromeCompression.CCIT4;

    options.optimization = false;
    options.viewAfterSaving = false;
    options.requireDocumentPassword = false;
    options.requirePermissionPassword = false;
    return options;
}

function sourceToTarget(sourceFile, outputFolder) {
    return new File(outputFolder.fsName + "/" + basenameWithoutExtension(sourceFile) + ".pdf");
}

function validateInputs(config, outputFolder) {
    var sources = [];
    var targets = [];
    var i, sourcePath, sourceFile, targetFile;

    for (i = 0; i < config.sourceFiles.length; i++) {
        sourcePath = config.sourceFiles[i];
        if (!isAbsolutePath(sourcePath)) {
            fail("sourceFiles[" + i + "] must be an absolute path: " + sourcePath);
        }
        if (!hasAiExtension(sourcePath)) {
            fail("sourceFiles[" + i + "] must end with .ai: " + sourcePath);
        }
        sourceFile = new File(sourcePath);
        if (!sourceFile.exists) {
            fail("Source file does not exist: " + sourceFile.fsName);
        }
        targetFile = sourceToTarget(sourceFile, outputFolder);
        if (targetFile.exists && !config.overwriteExisting) {
            fail("Target exists and overwriteExisting is false: " + targetFile.fsName);
        }
        sources.push(sourceFile);
        targets.push(targetFile);
    }
    return { sources: sources, targets: targets };
}

function exportFiles(config) {
    var logFile = openLog(config.logFile);
    var outputFolder = new Folder(config.outputDir);
    if (!outputFolder.exists && !outputFolder.create()) {
        fail("Cannot create outputDir: " + outputFolder.fsName);
    }

    var validated = validateInputs(config, outputFolder);
    var sources = validated.sources;
    var targets = validated.targets;
    var options = buildPdfOptions();
    var previousInteractionLevel = app.userInteractionLevel;
    var doc = null;
    var i;

    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;
    try {
        for (i = 0; i < sources.length; i++) {
            doc = null;
            log(logFile, "OPEN " + sources[i].fsName);
            doc = app.open(sources[i]);
            log(logFile, "SAVE " + targets[i].fsName);
            doc.saveAs(targets[i], options);
            log(logFile, "DONE " + targets[i].fsName + " size=" + targets[i].length);
            doc.close(SaveOptions.DONOTSAVECHANGES);
            doc = null;
        }
    } catch (err) {
        log(logFile, "FAIL " + err);
        if (doc !== null) {
            try {
                doc.close(SaveOptions.DONOTSAVECHANGES);
            } catch (closeErr) {
                log(logFile, "WARN close failed " + closeErr);
            }
        }
        fail(err);
    } finally {
        app.userInteractionLevel = previousInteractionLevel;
    }

    log(logFile, "Completed " + sources.length + " exports");
    return "Completed " + sources.length + " exports";
}

exportFiles(readConfig());
