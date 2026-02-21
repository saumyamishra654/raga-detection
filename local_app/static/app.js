(function () {
    const modeSelect = document.getElementById("mode-select");
    const schemaForms = document.getElementById("schema-forms");
    const extraArgsEl = document.getElementById("extra-args");
    const runBtn = document.getElementById("run-btn");
    const nextBtn = document.getElementById("next-btn");
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
    let isBusy = false;
    let selectedAudioPath = "";
    let selectedAudioDirectory = "";
    let pendingNextJobId = null;
    let pendingNextSuggestion = null;
    const modeDrafts = new Map();
    let currentReportLinks = { detect: null, analyze: null };
    const ragaNameCacheBySource = new Map();
    const audioFileCacheBySource = new Map();
    const tanpuraTrackCache = new Map();
    let activeRecordingContext = null;
    const AUDIO_DIR_STORAGE_KEY = "ragaLocalApp.audioDirectory";
    const DEFAULT_AUDIO_DIR_REL = (document.body && document.body.dataset && document.body.dataset.defaultAudioDir)
        ? document.body.dataset.defaultAudioDir
        : "../audio_test_files";
    const DEFAULT_MODE = (document.body && document.body.dataset && document.body.dataset.defaultMode)
        ? String(document.body.dataset.defaultMode).trim()
        : "detect";

    const FIELD_DEPENDENCIES = {
        vocalist_gender: { field: "source_type", equals: "vocal" },
        instrument_type: { field: "source_type", equals: "instrumental" },
        yt: { field: "ingest", equals: "youtube" },
        start_time: { field: "ingest", equals: "youtube" },
        end_time: { field: "ingest", equals: "youtube" },
        record_mode: { field: "ingest", equals: "record" },
        recorded_audio: { field: "ingest", equals: "record" },
        tanpura_key: { field: "record_mode", equals: "tanpura_vocal" }
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

    async function fetchTanpuraTracks(forceRefresh) {
        const sourceKey = "__tanpura__";
        if (!forceRefresh && tanpuraTrackCache.has(sourceKey)) {
            return tanpuraTrackCache.get(sourceKey);
        }
        const res = await fetch("/api/tanpura-tracks");
        if (!res.ok) {
            throw new Error("Failed to load tanpura tracks");
        }
        const payload = await res.json();
        tanpuraTrackCache.set(sourceKey, payload);
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

    function getFieldInput(name) {
        return document.getElementById(fieldInputId(name));
    }

    function getFieldValue(name) {
        const input = getFieldInput(name);
        if (!input) return "";
        return String(input.value || "").trim();
    }

    function composePreprocessOutputAudioPath() {
        const audioDir = getFieldValue("audio_dir") || getFieldDefault("audio_dir", DEFAULT_AUDIO_DIR_REL) || "";
        const filename = getFieldValue("filename");
        if (!filename) return "";

        const normalizedDir = String(audioDir).trim().replace(/[\\/]+$/, "");
        if (!normalizedDir) {
            return filename + ".mp3";
        }
        return normalizedDir + "/" + filename + ".mp3";
    }

    function buildRecordDetectSuggestion() {
        if (modeSelect.value !== "preprocess") return null;
        const ingest = getFieldValue("ingest") || "youtube";
        if (ingest !== "record") return null;

        const recordedPath = getFieldValue("recorded_audio");
        const outputAudioPath = recordedPath || composePreprocessOutputAudioPath();
        if (!outputAudioPath) return null;

        const paramsByFlag = {
            "--audio": outputAudioPath,
        };
        const consumedFlags = ["--audio"];

        const outputDir = getFieldValue("output") || getFieldDefault("output", "batch_results");
        if (outputDir) {
            paramsByFlag["--output"] = outputDir;
            consumedFlags.push("--output");
        }

        const recordMode = getFieldValue("record_mode") || "song";
        if (recordMode === "tanpura_vocal") {
            const tanpuraKey = getFieldValue("tanpura_key");
            if (tanpuraKey) {
                paramsByFlag["--tonic"] = tanpuraKey;
                consumedFlags.push("--tonic");
            }
            paramsByFlag["--skip-separation"] = true;
            consumedFlags.push("--skip-separation");
        }

        return { mode: "detect", paramsByFlag: paramsByFlag, consumedFlags: consumedFlags };
    }

    function getRecordingStartValidationError(tanpuraSelect) {
        if (modeSelect.value !== "preprocess") {
            return "Recording controls are available only in preprocess mode.";
        }

        const ingest = getFieldValue("ingest");
        if (ingest !== "record") {
            return "Set --ingest to 'record' to use microphone recording.";
        }

        const filename = getFieldValue("filename");
        if (!filename) {
            return "Fill --filename before starting recording.";
        }

        const recordMode = getFieldValue("record_mode") || "song";
        if (recordMode === "tanpura_vocal") {
            const selectedTanpura = String(
                (tanpuraSelect && tanpuraSelect.value) || getFieldValue("tanpura_key") || ""
            ).trim();
            if (!selectedTanpura) {
                return "Select a tanpura key before starting recording.";
            }
        }
        return "";
    }

    function validateClientSideSubmitOrThrow(mode, params) {
        if (mode === "detect") {
            const skipInput = getFieldInput("skip_separation");
            const skipEnabled = Boolean(
                params.skip_separation ||
                (skipInput && skipInput.checked)
            );
            if (skipEnabled) {
                const tonicRaw = String(params.tonic || getFieldValue("tonic") || "").trim();
                if (!tonicRaw) {
                    throw new Error("Missing required field: --tonic (required when --skip-separation is enabled).");
                }

                const melodyInput = getFieldInput("melody_source");
                const currentMelodySource = String(
                    params.melody_source ||
                    (melodyInput ? melodyInput.value : "") ||
                    "separated"
                ).trim();
                if (currentMelodySource !== "composite") {
                    if (melodyInput) {
                        melodyInput.value = "composite";
                    }
                    params.melody_source = "composite";
                }
            }
        }

        if (mode !== "preprocess") return;
        const ingest = String(params.ingest || getFieldValue("ingest") || "youtube").trim();
        if (ingest === "youtube") {
            const ytUrl = String(params.yt || getFieldValue("yt") || "").trim();
            if (!ytUrl) {
                throw new Error("Missing required field: --yt (required when --ingest is youtube).");
            }
        }
    }

    function enforceDetectSkipSeparationRules() {
        if (modeSelect.value !== "detect") return;
        const skipInput = getFieldInput("skip_separation");
        const melodyInput = getFieldInput("melody_source");
        if (!skipInput || !melodyInput) return;
        if (skipInput.checked && String(melodyInput.value || "").trim() !== "composite") {
            melodyInput.value = "composite";
        }
    }

    function isRecordingInProgress() {
        return Boolean(activeRecordingContext && activeRecordingContext.backendRecording);
    }

    function stopActiveRecording(cancelUpload) {
        if (!isRecordingInProgress()) return;
        if (!cancelUpload) return;
        if (!activeRecordingContext.cancelRequested) {
            activeRecordingContext.cancelRequested = true;
            fetch("/api/preprocess-record/cancel", { method: "POST" }).catch(function () {
                // Ignore cleanup errors on fire-and-forget cancellation.
            });
        }
        activeRecordingContext = null;
    }

    async function startBackendRecording(payload) {
        const res = await fetch("/api/preprocess-record/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!res.ok) {
            const body = await res.json().catch(function () { return {}; });
            throw new Error(body.detail || "Failed to start recording");
        }
        return res.json();
    }

    async function stopBackendRecording() {
        const res = await fetch("/api/preprocess-record/stop", { method: "POST" });
        if (!res.ok) {
            const body = await res.json().catch(function () { return {}; });
            throw new Error(body.detail || "Failed to stop recording");
        }
        return res.json();
    }

    async function cancelBackendRecording() {
        const res = await fetch("/api/preprocess-record/cancel", { method: "POST" });
        if (!res.ok) {
            const body = await res.json().catch(function () { return {}; });
            throw new Error(body.detail || "Failed to cancel recording");
        }
        return res.json();
    }

    function attachPreprocessRecordingControls(row, recordedAudioInput) {
        const helperWrap = document.createElement("div");
        helperWrap.className = "hint preprocess-recording-wrap";

        const controls = document.createElement("div");
        controls.className = "preprocess-recording-controls";

        const startBtn = document.createElement("button");
        startBtn.type = "button";
        startBtn.textContent = "Start Recording";

        const stopBtn = document.createElement("button");
        stopBtn.type = "button";
        stopBtn.textContent = "Stop Recording";
        stopBtn.disabled = true;

        const clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.textContent = "Clear Recording";

        controls.appendChild(startBtn);
        controls.appendChild(stopBtn);
        controls.appendChild(clearBtn);
        helperWrap.appendChild(controls);

        const tanpuraWrap = document.createElement("div");
        tanpuraWrap.className = "preprocess-tanpura-wrap";

        const tanpuraLabel = document.createElement("div");
        tanpuraLabel.className = "hint";
        tanpuraLabel.textContent = "Tanpura (for tanpura_vocal mode):";
        tanpuraWrap.appendChild(tanpuraLabel);

        const tanpuraControls = document.createElement("div");
        tanpuraControls.className = "preprocess-tanpura-controls";

        const tanpuraSelect = document.createElement("select");
        tanpuraSelect.className = "preprocess-tanpura-select";

        const tanpuraPlayBtn = document.createElement("button");
        tanpuraPlayBtn.type = "button";
        tanpuraPlayBtn.textContent = "Play Tanpura";

        const tanpuraStopBtn = document.createElement("button");
        tanpuraStopBtn.type = "button";
        tanpuraStopBtn.textContent = "Stop Tanpura";
        tanpuraStopBtn.disabled = true;

        tanpuraControls.appendChild(tanpuraSelect);
        tanpuraControls.appendChild(tanpuraPlayBtn);
        tanpuraControls.appendChild(tanpuraStopBtn);
        tanpuraWrap.appendChild(tanpuraControls);

        helperWrap.appendChild(tanpuraWrap);

        const preview = document.createElement("audio");
        preview.controls = true;
        preview.className = "preprocess-recording-preview";
        preview.style.display = "none";
        helperWrap.appendChild(preview);

        const status = document.createElement("div");
        status.className = "hint";
        status.textContent = "Use ffmpeg microphone recording for preprocess ingest.";
        helperWrap.appendChild(status);

        const tanpuraFieldInput = getFieldInput("tanpura_key");
        const tanpuraTrackByKey = new Map();
        let tanpuraPreviewAudio = null;

        function updateButtons(isRecording) {
            startBtn.disabled = isRecording;
            stopBtn.disabled = !isRecording;
        }

        function updateTanpuraPreviewButtons(isPlaying) {
            tanpuraPlayBtn.disabled = isPlaying;
            tanpuraStopBtn.disabled = !isPlaying;
        }

        function stopTanpuraPreview() {
            if (!tanpuraPreviewAudio) {
                updateTanpuraPreviewButtons(false);
                return;
            }
            try {
                tanpuraPreviewAudio.pause();
                tanpuraPreviewAudio.currentTime = 0;
            } catch (_err) {
                // Ignore preview cleanup errors.
            }
            tanpuraPreviewAudio = null;
            updateTanpuraPreviewButtons(false);
        }

        function syncTanpuraFieldFromSelect() {
            if (!tanpuraFieldInput) return;
            tanpuraFieldInput.value = String(tanpuraSelect.value || "").trim();
        }

        function syncTanpuraSelectFromField() {
            if (!tanpuraFieldInput) return;
            const key = String(tanpuraFieldInput.value || "").trim();
            if (key && tanpuraTrackByKey.has(key)) {
                tanpuraSelect.value = key;
            }
        }

        function setTanpuraVisibility() {
            const ingest = getFieldValue("ingest");
            const recordMode = getFieldValue("record_mode") || "song";
            const shouldShow = ingest === "record" && recordMode === "tanpura_vocal";
            tanpuraWrap.style.display = shouldShow ? "grid" : "none";
            if (!shouldShow) {
                stopTanpuraPreview();
            }
        }

        async function populateTanpuraSelect(forceRefresh) {
            const payload = await fetchTanpuraTracks(forceRefresh);
            const tracks = Array.isArray(payload.tracks) ? payload.tracks : [];
            tanpuraSelect.innerHTML = "";
            tanpuraTrackByKey.clear();

            const placeholder = document.createElement("option");
            placeholder.value = "";
            placeholder.textContent = "Select tanpura track...";
            tanpuraSelect.appendChild(placeholder);

            tracks.forEach(function (track) {
                if (!track || !track.key) return;
                tanpuraTrackByKey.set(String(track.key), track);
                const option = document.createElement("option");
                option.value = String(track.key);
                option.textContent = String(track.label || track.key);
                tanpuraSelect.appendChild(option);
            });

            const preferredKey = (tanpuraFieldInput && tanpuraFieldInput.value)
                ? String(tanpuraFieldInput.value).trim()
                : "";
            if (preferredKey && tanpuraTrackByKey.has(preferredKey)) {
                tanpuraSelect.value = preferredKey;
            }
        }

        async function resolveTanpuraUrlForCurrentSelection() {
            const tanpuraKey = String(tanpuraSelect.value || getFieldValue("tanpura_key") || "").trim();
            if (!tanpuraKey) {
                throw new Error("Select a tanpura key before recording.");
            }
            if (tanpuraFieldInput) {
                tanpuraFieldInput.value = tanpuraKey;
            }

            let track = tanpuraTrackByKey.get(tanpuraKey);
            if (!track) {
                const payload = await fetchTanpuraTracks(false);
                const tracks = Array.isArray(payload.tracks) ? payload.tracks : [];
                tracks.forEach(function (item) {
                    if (!item || !item.key) return;
                    tanpuraTrackByKey.set(String(item.key), item);
                });
                track = tanpuraTrackByKey.get(tanpuraKey);
            }
            if (!track || !track.url) {
                throw new Error("Selected tanpura track is not available.");
            }
            return track.url;
        }

        tanpuraSelect.addEventListener("change", function () {
            syncTanpuraFieldFromSelect();
            stopTanpuraPreview();
        });

        tanpuraPlayBtn.addEventListener("click", async function () {
            try {
                const tanpuraUrl = await resolveTanpuraUrlForCurrentSelection();
                stopTanpuraPreview();
                tanpuraPreviewAudio = new Audio(tanpuraUrl);
                tanpuraPreviewAudio.loop = true;
                await tanpuraPreviewAudio.play();
                updateTanpuraPreviewButtons(true);
                status.textContent = "Tanpura preview playing.";
            } catch (err) {
                status.textContent = "Tanpura preview error: " + err.message;
            }
        });

        tanpuraStopBtn.addEventListener("click", function () {
            stopTanpuraPreview();
            status.textContent = "Tanpura preview stopped.";
        });

        if (tanpuraFieldInput) {
            tanpuraFieldInput.addEventListener("change", function () {
                syncTanpuraSelectFromField();
                setTanpuraVisibility();
            });
        }

        ["ingest", "record_mode"].forEach(function (name) {
            const input = getFieldInput(name);
            if (!input) return;
            input.addEventListener("change", function () {
                setTanpuraVisibility();
            });
        });

        populateTanpuraSelect(true).catch(function (err) {
            status.textContent = "Tanpura list error: " + err.message;
        });
        syncTanpuraSelectFromField();
        setTanpuraVisibility();

        startBtn.addEventListener("click", async function () {
            if (isRecordingInProgress()) {
                status.textContent = "A recording is already in progress.";
                return;
            }

            const recordingValidationError = getRecordingStartValidationError(tanpuraSelect);
            if (recordingValidationError) {
                status.textContent = recordingValidationError;
                return;
            }

            const recordMode = getFieldValue("record_mode") || "song";
            let tanpuraKey = null;
            if (recordMode === "tanpura_vocal") {
                tanpuraKey = String(tanpuraSelect.value || getFieldValue("tanpura_key") || "").trim() || null;
            }
            if (tanpuraFieldInput) {
                tanpuraFieldInput.value = tanpuraKey || "";
            }

            stopTanpuraPreview();
            clearPendingNextSuggestion();
            updateNextButtonState();

            try {
                status.textContent = "Starting ffmpeg recording...";
                const payload = await startBackendRecording({
                    audio_dir: getFieldValue("audio_dir") || getFieldDefault("audio_dir", DEFAULT_AUDIO_DIR_REL),
                    filename: getFieldValue("filename"),
                    record_mode: recordMode,
                    tanpura_key: tanpuraKey,
                });
                if (tanpuraFieldInput && payload && payload.tonic) {
                    tanpuraFieldInput.value = String(payload.tonic);
                    syncTanpuraSelectFromField();
                }
                activeRecordingContext = {
                    backendRecording: true,
                    cancelRequested: false,
                };
                updateButtons(true);
                status.textContent = "Recording... click Stop Recording to finish.";
                const deviceLabel = payload && payload.audio_input_device
                    ? (" (input " + payload.audio_input_device + ")")
                    : "";
                setStatus("Recording started using ffmpeg microphone capture" + deviceLabel + ".");
            } catch (err) {
                activeRecordingContext = null;
                updateButtons(false);
                status.textContent = "Recording start error: " + err.message;
                setStatus("Recording start error: " + err.message);
            }
        });

        stopBtn.addEventListener("click", async function () {
            if (!isRecordingInProgress()) return;
            status.textContent = "Stopping recording...";
            stopBtn.disabled = true;
            try {
                const payload = await stopBackendRecording();
                activeRecordingContext = null;
                updateButtons(false);

                recordedAudioInput.value = payload.path || "";
                if (recordedAudioInput.value) {
                    recordedAudioInput.classList.add("audio-path-hidden");
                    onAudioSelectionChanged(recordedAudioInput.value);
                } else {
                    recordedAudioInput.classList.remove("audio-path-hidden");
                }

                if (payload.url) {
                    preview.src = payload.url;
                    preview.style.display = "block";
                }

                if (tanpuraFieldInput && payload.tonic) {
                    tanpuraFieldInput.value = String(payload.tonic);
                    syncTanpuraSelectFromField();
                }

                status.textContent = "Recorded audio ready: " + (payload.filename || payload.path || "");
                setStatus("Recorded audio saved and autofilled: " + (payload.filename || payload.path || ""));

                const suggestion = buildRecordDetectSuggestion();
                if (suggestion) {
                    pendingNextSuggestion = suggestion;
                    pendingNextJobId = null;
                    updateNextButtonState();
                    setStatus(
                        "Recorded audio saved. Click Next to load detect with " +
                        suggestion.paramsByFlag["--audio"] +
                        (suggestion.paramsByFlag["--tonic"] ? (" and tonic " + suggestion.paramsByFlag["--tonic"]) : "") +
                        "."
                    );
                }
            } catch (err) {
                activeRecordingContext = null;
                updateButtons(false);
                status.textContent = "Recording stop error: " + err.message;
                setStatus("Recording stop error: " + err.message);
            }
        });

        clearBtn.addEventListener("click", async function () {
            if (isRecordingInProgress()) {
                try {
                    await cancelBackendRecording();
                } catch (_err) {
                    // Ignore cancellation errors during explicit clear.
                }
                activeRecordingContext = null;
                updateButtons(false);
            }
            stopTanpuraPreview();
            recordedAudioInput.value = "";
            recordedAudioInput.classList.remove("audio-path-hidden");
            preview.pause();
            preview.removeAttribute("src");
            preview.style.display = "none";
            clearPendingNextSuggestion();
            updateNextButtonState();
            status.textContent = "Recorded audio cleared.";
            setStatus("Cleared recorded audio path.");
        });

        if (recordedAudioInput.value) {
            recordedAudioInput.classList.add("audio-path-hidden");
            status.textContent = "Using uploaded recording: " + recordedAudioInput.value;
        }

        row.appendChild(helperWrap);
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
        if (field.name === "audio" || field.name === "recorded_audio") {
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
                if (field.name === "recorded_audio") {
                    attachPreprocessRecordingControls(row, input);
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
        ["source_type", "ingest", "record_mode"].forEach(function (name) {
            const input = document.getElementById(fieldInputId(name));
            if (!input) return;
            input.addEventListener("change", function () {
                updateConditionalVisibility();
            });
        });

        ["skip_separation", "melody_source"].forEach(function (name) {
            const input = document.getElementById(fieldInputId(name));
            if (!input) return;
            input.addEventListener("change", function () {
                enforceDetectSkipSeparationRules();
            });
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

        if (fieldName === "recorded_audio") {
            stopActiveRecording(true);
        }

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
        input.classList.remove("audio-path-hidden");
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
        enforceDetectSkipSeparationRules();
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

    function captureDraftForCurrentMode() {
        if (!currentSchema || !Array.isArray(currentSchema.fields)) return;
        const draft = {
            params: {},
            extraArgs: String(extraArgsEl.value || ""),
        };

        currentSchema.fields.forEach(function (field) {
            const input = document.getElementById(fieldInputId(field.name));
            if (!input) return;
            if (field.value_type === "bool") {
                draft.params[field.name] = Boolean(input.checked);
                return;
            }
            draft.params[field.name] = String(input.value || "");
        });

        modeDrafts.set(currentSchema.mode, draft);
    }

    function restoreDraftForCurrentMode() {
        if (!currentSchema) return;
        const draft = modeDrafts.get(currentSchema.mode);
        if (!draft) return;

        if (typeof draft.extraArgs === "string") {
            extraArgsEl.value = draft.extraArgs;
        }

        currentSchema.fields.forEach(function (field) {
            if (!Object.prototype.hasOwnProperty.call(draft.params, field.name)) return;
            const input = document.getElementById(fieldInputId(field.name));
            if (!input) return;

            const value = draft.params[field.name];
            if (field.value_type === "bool") {
                input.checked = Boolean(value);
                return;
            }
            input.value = value === null || value === undefined ? "" : String(value);
        });

        updateConditionalVisibility();
        const audioInput = document.getElementById(fieldInputId("audio"));
        if (audioInput && audioInput.value) {
            onAudioSelectionChanged(audioInput.value, { directory: selectedAudioDirectory });
        }
    }

    function clearPendingNextSuggestion() {
        pendingNextJobId = null;
        pendingNextSuggestion = null;
    }

    function updateNextButtonState() {
        if (!nextBtn) return;
        const modeAllowsNext = modeSelect.value === "preprocess" || modeSelect.value === "detect";
        const hasSuggestion = Boolean(pendingNextSuggestion) || Boolean(pendingNextJobId);
        const canAdvance = hasSuggestion && modeAllowsNext && !isBusy;
        nextBtn.disabled = !canAdvance;
        nextBtn.classList.toggle("ready", canAdvance);
    }

    function stripAnsi(text) {
        return String(text || "").replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, "");
    }

    function parseSuggestedCommand(logLines) {
        if (!Array.isArray(logLines) || !logLines.length) return null;
        const cleanedLines = logLines.map(function (line) {
            return stripAnsi(line).trim();
        });

        let startIdx = -1;
        for (let i = cleanedLines.length - 1; i >= 0; i--) {
            if (cleanedLines[i].includes("run_pipeline.sh")) {
                startIdx = i;
                break;
            }
        }
        if (startIdx < 0) return null;

        const commandParts = [];
        for (let i = startIdx; i < cleanedLines.length; i++) {
            const line = cleanedLines[i];
            if (!line) continue;
            if (i > startIdx && (line.startsWith("=") || line.startsWith("Next:"))) {
                break;
            }
            const normalized = line.replace(/\\$/, "").trim();
            if (!normalized) continue;
            commandParts.push(normalized);
        }
        if (!commandParts.length) return null;

        const commandText = commandParts.join(" ");
        const tokenRegex = /"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|(\S+)/g;
        const tokens = [];
        let match;
        while ((match = tokenRegex.exec(commandText)) !== null) {
            const token = match[1] !== undefined ? match[1] : (match[2] !== undefined ? match[2] : match[3]);
            if (token !== undefined && token !== null && token !== "") {
                tokens.push(token);
            }
        }
        if (!tokens.length) return null;

        let runIdx = -1;
        for (let i = 0; i < tokens.length; i++) {
            if (tokens[i] === "./run_pipeline.sh" || tokens[i].endsWith("/run_pipeline.sh")) {
                runIdx = i;
                break;
            }
        }
        if (runIdx < 0 || runIdx + 1 >= tokens.length) return null;

        const mode = tokens[runIdx + 1];
        if (!mode) return null;

        const paramsByFlag = {};
        const consumedFlags = [];
        for (let i = runIdx + 2; i < tokens.length; i++) {
            const token = tokens[i];
            if (!(token.startsWith("--") || token.startsWith("-"))) continue;

            let value = true;
            const next = i + 1 < tokens.length ? tokens[i + 1] : null;
            if (next && !(next.startsWith("--") || next.startsWith("-"))) {
                value = next;
                i += 1;
            }
            paramsByFlag[token] = value;
            consumedFlags.push(token);
        }

        if (!consumedFlags.length) {
            return null;
        }
        return { mode, paramsByFlag, consumedFlags };
    }

    async function applySuggestedCommand(suggestion) {
        if (!suggestion) return false;

        if (modeSelect.value !== suggestion.mode) {
            modeSelect.value = suggestion.mode;
            await loadSchema(suggestion.mode, { restoreDraft: false });
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

        extraArgsEl.value = unknownFlags.length > 0 ? unknownFlags.join(" ") : "";

        updateConditionalVisibility();
        const audioInput = document.getElementById(fieldInputId("audio"));
        if (audioInput && audioInput.value) {
            onAudioSelectionChanged(audioInput.value, { directory: selectedAudioDirectory });
        }
        captureDraftForCurrentMode();
        return true;
    }

    async function loadSchema(mode, options) {
        stopActiveRecording(true);
        const res = await fetch("/api/schema/" + mode);
        if (!res.ok) throw new Error("Failed to load schema for mode " + mode);
        currentSchema = await res.json();
        renderSchema(currentSchema);
        const shouldRestoreDraft = !options || options.restoreDraft !== false;
        if (shouldRestoreDraft) {
            restoreDraftForCurrentMode();
        }
        updateNextButtonState();
    }

    function setStatus(text) {
        statusLine.textContent = text;
    }

    function setBusy(busy) {
        isBusy = Boolean(busy);
        runBtn.disabled = isBusy;
        cancelBtn.disabled = !isBusy || !activeJobId;
        if (runBatchBtn) {
            runBatchBtn.disabled = isBusy;
        }
        updateNextButtonState();
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

    async function refreshJob(jobId, options) {
        const opts = options || {};
        const [jobRes, logRes] = await Promise.all([
            fetch("/api/jobs/" + jobId),
            fetch("/api/jobs/" + jobId + "/logs"),
        ]);
        if (!jobRes.ok || !logRes.ok) {
            throw new Error("Failed to refresh job state");
        }

        const job = await jobRes.json();
        const logs = await logRes.json();
        const logLines = logs.logs || [];

        setStatus(job.status + " - " + (job.message || ""));
        progressEl.value = Number(job.progress || 0);
        logsEl.textContent = logLines.join("\n");
        jobArgvEl.textContent = job.argv && job.argv.length ? ("argv: " + job.argv.join(" ")) : "";
        const isRunning = job.status === "queued" || job.status === "running";
        const shouldRefreshArtifacts = Boolean(opts.refreshArtifacts) || !isRunning;
        if (shouldRefreshArtifacts) {
            renderArtifacts(job.artifacts || []);

            const reportLinks = deriveReportLinksFromArtifacts(job.artifacts || []);
            if (reportLinks.detectUrl || reportLinks.analyzeUrl) {
                applyReportLinks(reportLinks.detectUrl, reportLinks.analyzeUrl);
            } else if (selectedAudioPath) {
                loadArtifactsForSelectedAudio().catch(function () {
                    // Keep silent; status line already reflects job updates.
                });
            }
        }

        if (
            job.status === "completed" &&
            (job.mode === "preprocess" || job.mode === "detect")
        ) {
            pendingNextJobId = job.job_id;
            const preview = parseSuggestedCommand(logLines);
            if (preview) {
                setStatus("Completed. Click Next to load suggested " + preview.mode + " command.");
            } else {
                setStatus("Completed. Click Next to parse the suggested next command.");
            }
            updateNextButtonState();
        }

        if (isRunning) {
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
            refreshJob(jobId, { refreshArtifacts: false }).catch(function (err) {
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
            if (isRecordingInProgress()) {
                throw new Error("Stop recording before submitting the preprocess job.");
            }
            captureDraftForCurrentMode();
            const mode = modeSelect.value;
            enforceDetectSkipSeparationRules();
            const params = collectParamsOrThrow();
            validateClientSideSubmitOrThrow(mode, params);
            const payload = {
                mode: mode,
                params: params,
                extra_args: parseExtraArgs(extraArgsEl.value),
            };
            clearPendingNextSuggestion();
            updateNextButtonState();
            const job = await submitJob(payload, "/api/jobs");
            activeJobId = job.job_id;
            setBusy(true);
            setStatus("Submitted: " + job.job_id);
            logsEl.textContent = "";
            artifactListEl.textContent = "";
            applyReportLinks(null, null);
            startPolling(job.job_id);
            await refreshJob(job.job_id, { refreshArtifacts: false });
        } catch (err) {
            setStatus("Error: " + err.message);
        }
    });

    if (nextBtn) {
        nextBtn.addEventListener("click", async function () {
            if (isBusy) return;
            try {
                captureDraftForCurrentMode();
                let suggestion = pendingNextSuggestion;
                if (!suggestion) {
                    if (!pendingNextJobId) return;
                    const logRes = await fetch("/api/jobs/" + pendingNextJobId + "/logs");
                    if (!logRes.ok) {
                        throw new Error("Failed to fetch logs for Next transition.");
                    }
                    const logsPayload = await logRes.json();
                    suggestion = parseSuggestedCommand(logsPayload.logs || []);
                }
                if (!suggestion) {
                    setStatus("Next: no suggested command found yet. Wait for logs and retry.");
                    return;
                }
                const targetMode = suggestion.mode;
                const applied = await applySuggestedCommand(suggestion);
                if (!applied) return;
                clearPendingNextSuggestion();
                updateNextButtonState();
                setStatus(
                    "Loaded suggested " + targetMode + " command (" +
                    String(suggestion.consumedFlags.length) +
                    " flags). Review parameters, then run."
                );
            } catch (err) {
                setStatus("Next error: " + err.message);
            }
        });
    }

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
                captureDraftForCurrentMode();
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
                clearPendingNextSuggestion();
                updateNextButtonState();
                const job = await submitJob(payload, "/api/batch-jobs");
                activeJobId = job.job_id;
                setBusy(true);
                setStatus("Batch submitted: " + job.job_id);
                logsEl.textContent = "";
                artifactListEl.textContent = "";
                applyReportLinks(null, null);
                startPolling(job.job_id);
                await refreshJob(job.job_id, { refreshArtifacts: false });
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
        captureDraftForCurrentMode();
        loadSchema(modeSelect.value).catch(function (err) {
            setStatus("Schema load error: " + err.message);
        });
    });

    if (DEFAULT_MODE) {
        const hasDefaultModeOption = Array.from(modeSelect.options).some(function (option) {
            return option.value === DEFAULT_MODE;
        });
        if (hasDefaultModeOption) {
            modeSelect.value = DEFAULT_MODE;
        }
    }

    schemaForms.addEventListener("input", captureDraftForCurrentMode);
    schemaForms.addEventListener("change", captureDraftForCurrentMode);
    extraArgsEl.addEventListener("input", captureDraftForCurrentMode);
    extraArgsEl.addEventListener("change", captureDraftForCurrentMode);
    window.addEventListener("beforeunload", function () {
        stopActiveRecording(true);
    });

    loadSchema(modeSelect.value).catch(function (err) {
        setStatus("Schema load error: " + err.message);
    });
})();
