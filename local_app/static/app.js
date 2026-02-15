(function () {
    const modeSelect = document.getElementById("mode-select");
    const schemaForms = document.getElementById("schema-forms");
    const extraArgsEl = document.getElementById("extra-args");
    const runBtn = document.getElementById("run-btn");
    const rerunBtn = document.getElementById("rerun-btn");
    const cancelBtn = document.getElementById("cancel-btn");
    const statusLine = document.getElementById("status-line");
    const progressEl = document.getElementById("progress");
    const logsEl = document.getElementById("logs");
    const artifactListEl = document.getElementById("artifact-list");
    const jobArgvEl = document.getElementById("job-argv");
    const openDetectReportBtn = document.getElementById("open-detect-report-btn");
    const openAnalyzeReportBtn = document.getElementById("open-analyze-report-btn");
    const runBatchBtn = document.getElementById("run-batch-btn");

    let currentSchema = null;
    let activeJobId = null;
    let pollHandle = null;
    let lastPayload = null;
    let autoPrefilledFromJobId = null;
    let selectedAudioPath = "";
    let selectedAudioDirectory = "";
    let currentReportLinks = { detect: null, analyze: null };
    const ragaNameCacheBySource = new Map();
    const audioFileCacheBySource = new Map();
    const AUDIO_DIR_STORAGE_KEY = "ragaLocalApp.audioDirectory";
    const DEFAULT_AUDIO_DIR_REL = (document.body && document.body.dataset && document.body.dataset.defaultAudioDir)
        ? document.body.dataset.defaultAudioDir
        : "../audio_test_files";

    const FIELD_DEPENDENCIES = {
        vocalist_gender: { field: "source_type", equals: "vocal" },
        instrument_type: { field: "source_type", equals: "instrumental" },
        skip_separation: { field: "source_type", equals: "instrumental" }
    };
    selectedAudioDirectory = getSavedAudioDirectory() || DEFAULT_AUDIO_DIR_REL;

    function clearChildren(el) {
        while (el.firstChild) {
            el.removeChild(el.firstChild);
        }
    }

    function parseExtraArgs(value) {
        const trimmed = value.trim();
        if (!trimmed) return [];
        return trimmed.split(/\s+/);
    }

    function fieldInputId(fieldName) {
        return "field-" + fieldName.replace(/_/g, "-");
    }

    function getSavedAudioDirectory() {
        try {
            const value = window.localStorage.getItem(AUDIO_DIR_STORAGE_KEY);
            return value ? value.trim() : "";
        } catch (_err) {
            return "";
        }
    }

    function saveAudioDirectory(pathValue) {
        const value = String(pathValue || "").trim();
        if (!value) return;
        try {
            window.localStorage.setItem(AUDIO_DIR_STORAGE_KEY, value);
        } catch (_err) {
            // Ignore localStorage failures for private/incognito contexts.
        }
    }

    async function fetchAudioFiles(audioDirPath, forceRefresh) {
        const sourceKey = String(audioDirPath || "").trim() || DEFAULT_AUDIO_DIR_REL;
        if (!forceRefresh && audioFileCacheBySource.has(sourceKey)) {
            return audioFileCacheBySource.get(sourceKey);
        }

        const endpoint = new URL("/api/audio-files", window.location.origin);
        endpoint.searchParams.set("audio_dir", sourceKey);
        const res = await fetch(endpoint.toString());
        if (!res.ok) {
            throw new Error("Failed to load audio files");
        }
        const payload = await res.json();
        audioFileCacheBySource.set(sourceKey, payload);
        return payload;
    }

    function getFieldDefault(name, fallbackValue) {
        if (!currentSchema || !Array.isArray(currentSchema.fields)) return fallbackValue;
        const field = currentSchema.fields.find(function (item) { return item.name === name; });
        if (!field || field.default === null || field.default === undefined || field.default === "") {
            return fallbackValue;
        }
        return String(field.default);
    }

    function getCurrentOutputDirectory() {
        const outputInput = document.getElementById(fieldInputId("output"));
        if (outputInput && String(outputInput.value || "").trim()) {
            return String(outputInput.value || "").trim();
        }
        return getFieldDefault("output", "batch_results");
    }

    function getCurrentSeparator() {
        const separatorInput = document.getElementById(fieldInputId("separator"));
        if (separatorInput && String(separatorInput.value || "").trim()) {
            return String(separatorInput.value || "").trim();
        }
        return getFieldDefault("separator", "demucs");
    }

    function getCurrentDemucsModel() {
        const modelInput = document.getElementById(fieldInputId("demucs_model"));
        if (modelInput && String(modelInput.value || "").trim()) {
            return String(modelInput.value || "").trim();
        }
        return getFieldDefault("demucs_model", "htdemucs");
    }

    function applyReportLinks(detectUrl, analyzeUrl) {
        currentReportLinks = {
            detect: detectUrl || null,
            analyze: analyzeUrl || null,
        };
        if (openDetectReportBtn) {
            openDetectReportBtn.disabled = !currentReportLinks.detect;
        }
        if (openAnalyzeReportBtn) {
            openAnalyzeReportBtn.disabled = !currentReportLinks.analyze;
        }
    }

    async function fetchAudioArtifacts(audioPath) {
        const endpoint = new URL("/api/audio-artifacts", window.location.origin);
        endpoint.searchParams.set("audio_path", audioPath);
        endpoint.searchParams.set("output_dir", getCurrentOutputDirectory());
        endpoint.searchParams.set("separator", getCurrentSeparator());
        endpoint.searchParams.set("demucs_model", getCurrentDemucsModel());
        const res = await fetch(endpoint.toString());
        if (!res.ok) {
            const body = await res.json().catch(function () { return {}; });
            throw new Error(body.detail || "Failed to load artifacts for selected audio");
        }
        return res.json();
    }

    async function loadArtifactsForSelectedAudio() {
        if (!selectedAudioPath) {
            applyReportLinks(null, null);
            clearChildren(artifactListEl);
            return;
        }

        const payload = await fetchAudioArtifacts(selectedAudioPath);
        renderArtifacts(payload.artifacts || []);
        applyReportLinks(payload.detect_report_url, payload.analyze_report_url);
        if (payload.found) {
            setStatus("Loaded existing artifacts for " + (payload.audio_stem || "selected audio"));
        } else {
            setStatus("No existing artifacts found yet for selected audio.");
        }
    }

    function onAudioSelectionChanged(audioPath, options) {
        selectedAudioPath = String(audioPath || "").trim();
        if (selectedAudioPath && options && options.directory) {
            selectedAudioDirectory = options.directory;
        }
        loadArtifactsForSelectedAudio().catch(function (err) {
            setStatus("Artifact lookup error: " + err.message);
        });
    }

    async function fetchRagaNames(ragaDbPath) {
        const sourceKey = String(ragaDbPath || "").trim() || "__default__";
        if (ragaNameCacheBySource.has(sourceKey)) {
            return ragaNameCacheBySource.get(sourceKey);
        }

        const endpoint = new URL("/api/ragas", window.location.origin);
        if (sourceKey !== "__default__") {
            endpoint.searchParams.set("raga_db", sourceKey);
        }
        const res = await fetch(endpoint.toString());
        if (!res.ok) {
            throw new Error("Failed to load raga names");
        }
        const payload = await res.json();
        const ragas = Array.isArray(payload.ragas) ? payload.ragas : [];
        ragaNameCacheBySource.set(sourceKey, ragas);
        return ragas;
    }

    function normalizeName(text) {
        return String(text || "")
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, "")
            .trim();
    }

    function levenshteinDistance(a, b) {
        const s = normalizeName(a);
        const t = normalizeName(b);
        if (!s.length) return t.length;
        if (!t.length) return s.length;

        const dp = Array.from({ length: s.length + 1 }, function () {
            return new Array(t.length + 1).fill(0);
        });
        for (let i = 0; i <= s.length; i++) dp[i][0] = i;
        for (let j = 0; j <= t.length; j++) dp[0][j] = j;

        for (let i = 1; i <= s.length; i++) {
            for (let j = 1; j <= t.length; j++) {
                const cost = s[i - 1] === t[j - 1] ? 0 : 1;
                dp[i][j] = Math.min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                );
            }
        }
        return dp[s.length][t.length];
    }

    function fuzzyBestRaga(query, ragas) {
        const q = normalizeName(query);
        if (!q || !ragas.length) return null;

        let best = null;
        let bestScore = -1;
        for (const candidate of ragas) {
            const c = normalizeName(candidate);
            if (!c) continue;
            let score = 0;
            if (q === c) score = 1.0;
            else if (c.startsWith(q) || q.startsWith(c)) score = 0.92;
            else if (c.includes(q) || q.includes(c)) score = 0.86;
            else {
                const dist = levenshteinDistance(q, c);
                score = 1 - dist / Math.max(q.length, c.length);
            }
            if (score > bestScore) {
                bestScore = score;
                best = candidate;
            }
        }
        if (bestScore >= 0.62) return best;
        return null;
    }

    async function uploadAudioFile(file, targetInput, dropzone) {
        if (!file) return;
        const originalLabel = dropzone.textContent;
        dropzone.textContent = "Uploading " + file.name + "...";
        dropzone.classList.add("uploading");

        try {
            const formData = new FormData();
            formData.append("audio_file", file, file.name);
            const res = await fetch("/api/upload-audio", {
                method: "POST",
                body: formData
            });
            if (!res.ok) {
                const body = await res.json().catch(function () { return {}; });
                throw new Error(body.detail || "Upload failed");
            }

            const payload = await res.json();
            targetInput.value = payload.path;
            targetInput.classList.add("audio-path-hidden");
            dropzone.textContent = "Uploaded: " + payload.filename;
            setStatus("Audio uploaded and autofilled: " + payload.filename);
            onAudioSelectionChanged(payload.path);
        } catch (err) {
            dropzone.textContent = originalLabel;
            setStatus("Audio upload error: " + err.message);
        } finally {
            dropzone.classList.remove("uploading");
        }
    }

    function attachAudioDropzone(row, audioInput) {
        const helperWrap = document.createElement("div");
        helperWrap.className = "hint audio-upload-wrap";

        const dropzone = document.createElement("div");
        dropzone.className = "audio-dropzone";
        dropzone.textContent = "Drag & drop audio file here, or click to browse";

        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = ".mp3,.wav,.flac,.m4a,.mp4,.aac,.ogg,audio/*";
        picker.style.display = "none";

        dropzone.addEventListener("click", function () {
            picker.click();
        });

        picker.addEventListener("change", function () {
            if (!picker.files || picker.files.length === 0) return;
            uploadAudioFile(picker.files[0], audioInput, dropzone);
        });

        ["dragenter", "dragover"].forEach(function (eventName) {
            dropzone.addEventListener(eventName, function (event) {
                event.preventDefault();
                dropzone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(function (eventName) {
            dropzone.addEventListener(eventName, function (event) {
                event.preventDefault();
                dropzone.classList.remove("dragover");
            });
        });

        dropzone.addEventListener("drop", function (event) {
            const files = event.dataTransfer ? event.dataTransfer.files : null;
            if (!files || files.length === 0) return;
            uploadAudioFile(files[0], audioInput, dropzone);
        });

        helperWrap.appendChild(dropzone);
        helperWrap.appendChild(picker);
        row.appendChild(helperWrap);
    }

    function attachAudioDirectoryPicker(row, audioInput) {
        const helperWrap = document.createElement("div");
        helperWrap.className = "hint audio-library-wrap";

        const controls = document.createElement("div");
        controls.className = "audio-library-controls";

        const dirInput = document.createElement("input");
        dirInput.type = "text";
        dirInput.className = "audio-dir-input";
        dirInput.placeholder = "Audio directory";
        dirInput.value = getSavedAudioDirectory() || DEFAULT_AUDIO_DIR_REL;

        const refreshBtn = document.createElement("button");
        refreshBtn.type = "button";
        refreshBtn.className = "audio-dir-refresh";
        refreshBtn.textContent = "Refresh";

        controls.appendChild(dirInput);
        controls.appendChild(refreshBtn);
        helperWrap.appendChild(controls);

        const fileSelect = document.createElement("select");
        fileSelect.className = "audio-file-select";
        helperWrap.appendChild(fileSelect);

        const info = document.createElement("div");
        info.className = "hint";
        helperWrap.appendChild(info);

        function setSelectOptions(payload) {
            fileSelect.innerHTML = "";
            let matchedExistingValue = false;

            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = "Select audio file from directory...";
            placeholder.selected = true;
            fileSelect.appendChild(placeholder);

            if (!payload.exists) {
                info.textContent = "Directory not found: " + payload.directory;
                return;
            }

            const files = Array.isArray(payload.files) ? payload.files : [];
            if (!files.length) {
                info.textContent = "No audio files found in: " + payload.directory;
                return;
            }

            files.forEach(function (item) {
                const option = document.createElement("option");
                option.value = String(item.path || "");
                option.textContent = String(item.name || item.path || "");
                if (audioInput.value && audioInput.value === option.value) {
                    option.selected = true;
                    placeholder.selected = false;
                    matchedExistingValue = true;
                }
                fileSelect.appendChild(option);
            });
            info.textContent = files.length + " audio file(s) found in: " + payload.directory;

            if (matchedExistingValue && audioInput.value) {
                onAudioSelectionChanged(audioInput.value, { directory: selectedAudioDirectory });
            }
        }

        async function refreshAudioFileOptions(forceRefresh) {
            const entered = dirInput.value.trim() || DEFAULT_AUDIO_DIR_REL;
            dirInput.value = entered;
            saveAudioDirectory(entered);
            const payload = await fetchAudioFiles(entered, forceRefresh);
            selectedAudioDirectory = payload.directory || entered;
            setSelectOptions(payload);
        }

        fileSelect.addEventListener("change", function () {
            if (!fileSelect.value) return;
            audioInput.value = fileSelect.value;
            audioInput.classList.add("audio-path-hidden");
            const chosen = fileSelect.options[fileSelect.selectedIndex];
            if (chosen) {
                setStatus("Audio selected: " + chosen.textContent);
            }
            onAudioSelectionChanged(fileSelect.value, { directory: selectedAudioDirectory });
        });

        refreshBtn.addEventListener("click", function () {
            refreshAudioFileOptions(true).catch(function (err) {
                setStatus("Audio directory error: " + err.message);
            });
        });

        dirInput.addEventListener("change", function () {
            refreshAudioFileOptions(true).catch(function (err) {
                setStatus("Audio directory error: " + err.message);
            });
        });

        dirInput.addEventListener("blur", function () {
            refreshAudioFileOptions(false).catch(function (err) {
                setStatus("Audio directory error: " + err.message);
            });
        });

        row.appendChild(helperWrap);
        audioInput.addEventListener("change", function () {
            onAudioSelectionChanged(audioInput.value, { directory: selectedAudioDirectory });
        });
        audioInput.addEventListener("blur", function () {
            if (audioInput.value) {
                onAudioSelectionChanged(audioInput.value, { directory: selectedAudioDirectory });
            }
        });
        refreshAudioFileOptions(true).catch(function (err) {
            setStatus("Audio directory error: " + err.message);
        });
    }

    async function attachRagaAutocomplete(row, ragaInput) {
        try {
            const listId = "raga-options-datalist";
            let datalist = document.getElementById(listId);
            if (!datalist) {
                datalist = document.createElement("datalist");
                datalist.id = listId;
                document.body.appendChild(datalist);
            }
            ragaInput.setAttribute("list", listId);

            let ragas = [];
            async function refreshRagaOptions() {
                const ragaDbInput = document.getElementById(fieldInputId("raga_db"));
                const ragaDbPath = ragaDbInput ? ragaDbInput.value.trim() : "";
                ragas = await fetchRagaNames(ragaDbPath);
                datalist.innerHTML = "";
                ragas.forEach(function (name) {
                    const option = document.createElement("option");
                    option.value = name;
                    datalist.appendChild(option);
                });
            }

            await refreshRagaOptions();
            const ragaDbInput = document.getElementById(fieldInputId("raga_db"));
            if (ragaDbInput) {
                const refreshWithErrorHandling = function () {
                    refreshRagaOptions().catch(function (err) {
                        setStatus("Raga list warning: " + err.message);
                    });
                };
                ragaDbInput.addEventListener("change", refreshWithErrorHandling);
                ragaDbInput.addEventListener("blur", refreshWithErrorHandling);
            }
            ragaInput.addEventListener("focus", function () {
                refreshRagaOptions().catch(function (err) {
                    setStatus("Raga list warning: " + err.message);
                });
            });

            const fuzzyHint = document.createElement("div");
            fuzzyHint.className = "hint";
            fuzzyHint.textContent = "Type freely; closest known raga will be auto-corrected on blur when confidence is high.";
            row.appendChild(fuzzyHint);

            ragaInput.addEventListener("blur", function () {
                const typed = ragaInput.value.trim();
                if (!typed) return;
                const exact = ragas.find(function (name) {
                    return normalizeName(name) === normalizeName(typed);
                });
                if (exact) {
                    ragaInput.value = exact;
                    return;
                }

                const match = fuzzyBestRaga(typed, ragas);
                if (match && normalizeName(match) !== normalizeName(typed)) {
                    ragaInput.value = match;
                    setStatus("Raga autocorrect: \"" + typed + "\" -> \"" + match + "\"");
                }
            });
        } catch (err) {
            setStatus("Raga list warning: " + err.message);
        }
    }

    function createInput(field) {
        const id = fieldInputId(field.name);
        let input;

        if (field.choices && field.choices.length > 0) {
            input = document.createElement("select");
            if (!field.required) {
                const blank = document.createElement("option");
                blank.value = "";
                blank.textContent = field.default !== null && field.default !== undefined
                    ? "(none: use default " + String(field.default) + ")"
                    : "(none)";
                blank.selected = true;
                input.appendChild(blank);
            }
            field.choices.forEach(function (choice) {
                const option = document.createElement("option");
                option.value = String(choice);
                option.textContent = String(choice);
                if (field.required && field.default !== null && String(field.default) === String(choice)) {
                    option.selected = true;
                }
                input.appendChild(option);
            });
        } else if (field.value_type === "bool") {
            input = document.createElement("input");
            input.type = "checkbox";
            const defaultBool = field.default === true;
            input.checked = defaultBool;
        } else {
            input = document.createElement("input");
            if (field.value_type === "int" || field.value_type === "float") {
                input.type = "number";
                if (field.value_type === "float") {
                    input.step = "any";
                } else {
                    input.step = "1";
                }
            } else {
                input.type = "text";
            }
            if (field.required && field.default !== null && field.default !== undefined) {
                input.value = String(field.default);
            } else if (!field.required && field.default !== null && field.default !== undefined) {
                input.placeholder = "Default: " + String(field.default);
            }
            if (field.required) {
                input.required = true;
            }
        }

        input.id = id;
        input.dataset.fieldName = field.name;
        if (field.name === "audio") {
            input.classList.add("audio-path-input");
        }
        return input;
    }

    function renderSchema(schema) {
        clearChildren(schemaForms);

        const groups = ["common", "mode", "advanced"];
        groups.forEach(function (group) {
            const fields = schema.fields.filter(function (f) {
                return f.group === group;
            });
            if (!fields.length) return;

            const section = document.createElement("section");
            section.className = "subsection";

            const title = document.createElement("h3");
            title.textContent = group === "mode" ? (schema.mode + " parameters") : group;
            section.appendChild(title);

            fields.forEach(function (field) {
                const row = document.createElement("div");
                row.className = "row";
                row.dataset.fieldName = field.name;

                const label = document.createElement("label");
                label.htmlFor = fieldInputId(field.name);
                label.textContent = field.flag + (field.required ? " *" : "");
                row.appendChild(label);

                const input = createInput(field);
                row.appendChild(input);

                const hint = document.createElement("div");
                hint.className = "hint";
                hint.textContent = field.help || "";
                row.appendChild(hint);

                if (field.name === "audio") {
                    attachAudioDirectoryPicker(row, input);
                    attachAudioDropzone(row, input);
                }
                if (field.name === "raga") {
                    attachRagaAutocomplete(row, input);
                }

                section.appendChild(row);
            });

            schemaForms.appendChild(section);
        });

        bindConditionalVisibilityHandlers();
        updateConditionalVisibility();
        bindArtifactContextHandlers();
    }

    function bindConditionalVisibilityHandlers() {
        const sourceInput = document.getElementById(fieldInputId("source_type"));
        if (!sourceInput) return;
        sourceInput.addEventListener("change", function () {
            updateConditionalVisibility();
        });
    }

    function bindArtifactContextHandlers() {
        ["output", "separator", "demucs_model"].forEach(function (name) {
            const input = document.getElementById(fieldInputId(name));
            if (!input) return;
            input.addEventListener("change", function () {
                if (!selectedAudioPath) return;
                loadArtifactsForSelectedAudio().catch(function (err) {
                    setStatus("Artifact lookup error: " + err.message);
                });
            });
            input.addEventListener("blur", function () {
                if (!selectedAudioPath) return;
                loadArtifactsForSelectedAudio().catch(function (err) {
                    setStatus("Artifact lookup error: " + err.message);
                });
            });
        });
    }

    function resetHiddenField(fieldName) {
        const field = currentSchema.fields.find(function (x) { return x.name === fieldName; });
        const input = document.getElementById(fieldInputId(fieldName));
        if (!field || !input) return;

        if (field.value_type === "bool") {
            if (field.action === "store_true") {
                input.checked = false;
            } else if (field.action === "store_false") {
                input.checked = true;
            } else {
                input.checked = false;
            }
            return;
        }
        input.value = "";
    }

    function updateConditionalVisibility() {
        Object.keys(FIELD_DEPENDENCIES).forEach(function (targetField) {
            const row = schemaForms.querySelector('.row[data-field-name="' + targetField + '"]');
            if (!row) return;

            const dep = FIELD_DEPENDENCIES[targetField];
            const sourceInput = document.getElementById(fieldInputId(dep.field));
            const shouldShow = Boolean(sourceInput) && sourceInput.value === dep.equals;
            row.style.display = shouldShow ? "grid" : "none";
            if (!shouldShow) {
                resetHiddenField(targetField);
            }
        });
    }

    function normalizeFieldValue(field, inputEl) {
        if (field.value_type === "bool") {
            return Boolean(inputEl.checked);
        }
        if (inputEl.value === "") {
            return null;
        }
        if (field.value_type === "int") {
            const parsed = parseInt(inputEl.value, 10);
            if (Number.isNaN(parsed)) throw new Error("Invalid integer value for " + field.flag);
            return parsed;
        }
        if (field.value_type === "float") {
            const parsed = parseFloat(inputEl.value);
            if (Number.isNaN(parsed)) throw new Error("Invalid numeric value for " + field.flag);
            return parsed;
        }
        return inputEl.value;
    }

    function collectParamsOrThrow() {
        if (!currentSchema) throw new Error("Schema not loaded");
        const params = {};

        currentSchema.fields.forEach(function (field) {
            const input = document.getElementById(fieldInputId(field.name));
            if (!input) return;

            const value = normalizeFieldValue(field, input);
            if (field.required && (value === null || value === "")) {
                throw new Error("Missing required field: " + field.flag);
            }

            if (field.value_type === "bool") {
                params[field.name] = value;
                return;
            }
            if (value !== null && value !== "") {
                params[field.name] = value;
            }
        });

        return params;
    }

    function parseSuggestedCommand(logLines) {
        if (!Array.isArray(logLines) || !logLines.length) return null;
        let startIdx = -1;
        for (let i = logLines.length - 1; i >= 0; i--) {
            if (logLines[i].includes("./run_pipeline.sh ")) {
                startIdx = i;
                break;
            }
        }
        if (startIdx < 0) return null;

        const firstLine = logLines[startIdx].trim().replace(/\\$/, "").trim();
        const firstMatch = firstLine.match(/\.\/run_pipeline\.sh\s+([a-zA-Z0-9_-]+)/);
        if (!firstMatch) return null;
        const mode = firstMatch[1];

        const paramsByFlag = {};
        const consumedFlags = [];
        for (let i = startIdx + 1; i < logLines.length; i++) {
            const line = logLines[i].trim();
            if (!line || line.startsWith("=") || line.startsWith("Next:")) {
                if (line.startsWith("=")) break;
                continue;
            }
            if (!line.startsWith("--")) continue;

            const clean = line.replace(/\\$/, "").trim();
            const match = clean.match(/^(--[a-zA-Z0-9-]+)(?:\s+"([^"]*)"|\s+(\S+))?$/);
            if (!match) continue;
            const flag = match[1];
            const value = match[2] !== undefined ? match[2] : (match[3] !== undefined ? match[3] : true);
            paramsByFlag[flag] = value;
            consumedFlags.push(flag);
        }

        return { mode, paramsByFlag, consumedFlags };
    }

    async function applySuggestedCommandFromLogs(logLines) {
        const suggestion = parseSuggestedCommand(logLines);
        if (!suggestion) return false;

        if (modeSelect.value !== suggestion.mode) {
            modeSelect.value = suggestion.mode;
            await loadSchema(suggestion.mode);
        }

        const flagToField = {};
        currentSchema.fields.forEach(function (field) {
            (field.flags || []).forEach(function (flag) {
                flagToField[flag] = field;
            });
        });

        const unknownFlags = [];
        Object.keys(suggestion.paramsByFlag).forEach(function (flag) {
            const field = flagToField[flag];
            if (!field) {
                unknownFlags.push(flag);
                return;
            }
            const input = document.getElementById(fieldInputId(field.name));
            if (!input) return;

            if (field.value_type === "bool") {
                if (field.action === "store_true") {
                    input.checked = true;
                } else if (field.action === "store_false") {
                    input.checked = false;
                }
                return;
            }

            const value = suggestion.paramsByFlag[flag];
            input.value = value === true ? "" : String(value);
        });

        if (unknownFlags.length > 0) {
            extraArgsEl.value = unknownFlags.join(" ");
        }

        updateConditionalVisibility();
        const audioInput = document.getElementById(fieldInputId("audio"));
        if (audioInput && audioInput.value) {
            onAudioSelectionChanged(audioInput.value, { directory: selectedAudioDirectory });
        }
        return true;
    }

    async function loadSchema(mode) {
        const res = await fetch("/api/schema/" + mode);
        if (!res.ok) throw new Error("Failed to load schema for mode " + mode);
        currentSchema = await res.json();
        renderSchema(currentSchema);
    }

    function setStatus(text) {
        statusLine.textContent = text;
    }

    function setBusy(isBusy) {
        runBtn.disabled = isBusy;
        cancelBtn.disabled = !isBusy || !activeJobId;
        if (runBatchBtn) {
            runBatchBtn.disabled = isBusy;
        }
    }

    function deriveReportLinksFromArtifacts(artifacts) {
        let detectUrl = null;
        let analyzeUrl = null;
        (artifacts || []).forEach(function (artifact) {
            if (!artifact || !artifact.url || !artifact.name) return;
            const lowerName = String(artifact.name).toLowerCase();
            if (!detectUrl && lowerName === "detection_report.html") {
                detectUrl = artifact.url;
            }
            if (!analyzeUrl && (lowerName === "analysis_report.html" || lowerName === "report.html")) {
                analyzeUrl = artifact.url;
            }
        });
        return { detectUrl: detectUrl, analyzeUrl: analyzeUrl };
    }

    function renderArtifacts(artifacts) {
        clearChildren(artifactListEl);

        (artifacts || []).forEach(function (artifact) {
            const li = document.createElement("li");
            if (artifact.url) {
                const a = document.createElement("a");
                a.href = artifact.url;
                a.textContent = artifact.name;
                a.target = "_blank";
                li.appendChild(a);
            } else {
                li.textContent = artifact.name + " (" + artifact.path + ")";
            }
            artifactListEl.appendChild(li);
        });
    }

    async function refreshJob(jobId) {
        const [jobRes, logRes] = await Promise.all([
            fetch("/api/jobs/" + jobId),
            fetch("/api/jobs/" + jobId + "/logs"),
        ]);
        if (!jobRes.ok || !logRes.ok) {
            throw new Error("Failed to refresh job state");
        }

        const job = await jobRes.json();
        const logs = await logRes.json();

        setStatus(job.status + " - " + (job.message || ""));
        progressEl.value = Number(job.progress || 0);
        logsEl.textContent = (logs.logs || []).join("\n");
        jobArgvEl.textContent = job.argv && job.argv.length ? ("argv: " + job.argv.join(" ")) : "";
        renderArtifacts(job.artifacts || []);

        const reportLinks = deriveReportLinksFromArtifacts(job.artifacts || []);
        if (reportLinks.detectUrl || reportLinks.analyzeUrl) {
            applyReportLinks(reportLinks.detectUrl, reportLinks.analyzeUrl);
        } else if (selectedAudioPath) {
            loadArtifactsForSelectedAudio().catch(function () {
                // Keep silent; status line already reflects job updates.
            });
        }

        if (job.status === "completed" && job.mode !== "batch" && autoPrefilledFromJobId !== job.job_id) {
            const applied = await applySuggestedCommandFromLogs(logs.logs || []);
            if (applied) {
                autoPrefilledFromJobId = job.job_id;
                setStatus("Completed. Loaded suggested " + modeSelect.value + " command into form.");
            }
        }

        if (job.status === "queued" || job.status === "running") {
            setBusy(true);
            cancelBtn.disabled = false;
        } else {
            setBusy(false);
            cancelBtn.disabled = true;
            activeJobId = null;
            if (pollHandle) {
                clearInterval(pollHandle);
                pollHandle = null;
            }
        }
    }

    function startPolling(jobId) {
        if (pollHandle) clearInterval(pollHandle);
        pollHandle = setInterval(function () {
            refreshJob(jobId).catch(function (err) {
                setStatus("Polling error: " + err.message);
            });
        }, 1000);
    }

    async function submitJob(payload, endpoint) {
        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            const body = await res.json().catch(function () { return {}; });
            throw new Error(body.detail || "Job submit failed");
        }
        return res.json();
    }

    runBtn.addEventListener("click", async function () {
        try {
            const mode = modeSelect.value;
            const params = collectParamsOrThrow();
            const payload = {
                mode: mode,
                params: params,
                extra_args: parseExtraArgs(extraArgsEl.value),
            };
            const job = await submitJob(payload, "/api/jobs");
            lastPayload = payload;
            rerunBtn.disabled = false;
            activeJobId = job.job_id;
            setBusy(true);
            setStatus("Submitted: " + job.job_id);
            logsEl.textContent = "";
            artifactListEl.textContent = "";
            applyReportLinks(null, null);
            startPolling(job.job_id);
            await refreshJob(job.job_id);
        } catch (err) {
            setStatus("Error: " + err.message);
        }
    });

    rerunBtn.addEventListener("click", async function () {
        if (!lastPayload) return;
        try {
            const job = await submitJob(lastPayload, "/api/jobs");
            activeJobId = job.job_id;
            setBusy(true);
            setStatus("Rerun submitted: " + job.job_id);
            logsEl.textContent = "";
            artifactListEl.textContent = "";
            applyReportLinks(null, null);
            startPolling(job.job_id);
            await refreshJob(job.job_id);
        } catch (err) {
            setStatus("Error: " + err.message);
        }
    });

    if (openDetectReportBtn) {
        openDetectReportBtn.addEventListener("click", function () {
            if (!currentReportLinks.detect) return;
            window.open(currentReportLinks.detect, "_blank", "noopener");
        });
    }

    if (openAnalyzeReportBtn) {
        openAnalyzeReportBtn.addEventListener("click", function () {
            if (!currentReportLinks.analyze) return;
            window.open(currentReportLinks.analyze, "_blank", "noopener");
        });
    }

    if (runBatchBtn) {
        runBatchBtn.addEventListener("click", async function () {
            try {
                const inputDir = (selectedAudioDirectory || getSavedAudioDirectory() || DEFAULT_AUDIO_DIR_REL).trim();
                if (!inputDir) {
                    throw new Error("Select or enter an audio directory first.");
                }
                const payload = {
                    input_dir: inputDir,
                    output_dir: getCurrentOutputDirectory(),
                    mode: "auto",
                    silent: true,
                };
                const job = await submitJob(payload, "/api/batch-jobs");
                activeJobId = job.job_id;
                setBusy(true);
                setStatus("Batch submitted: " + job.job_id);
                logsEl.textContent = "";
                artifactListEl.textContent = "";
                applyReportLinks(null, null);
                startPolling(job.job_id);
                await refreshJob(job.job_id);
            } catch (err) {
                setStatus("Batch error: " + err.message);
            }
        });
    }

    cancelBtn.addEventListener("click", async function () {
        if (!activeJobId) return;
        try {
            const res = await fetch("/api/jobs/" + activeJobId + "/cancel", { method: "POST" });
            if (!res.ok) {
                const body = await res.json().catch(function () { return {}; });
                throw new Error(body.detail || "Cancel failed");
            }
            await refreshJob(activeJobId);
        } catch (err) {
            setStatus("Cancel error: " + err.message);
        }
    });

    modeSelect.addEventListener("change", function () {
        loadSchema(modeSelect.value).catch(function (err) {
            setStatus("Schema load error: " + err.message);
        });
    });

    loadSchema(modeSelect.value).catch(function (err) {
        setStatus("Schema load error: " + err.message);
    });
})();
