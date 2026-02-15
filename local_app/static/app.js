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
    const reportFrame = document.getElementById("report-frame");
    const jobArgvEl = document.getElementById("job-argv");

    let currentSchema = null;
    let activeJobId = null;
    let pollHandle = null;
    let lastPayload = null;
    let autoPrefilledFromJobId = null;

    const FIELD_DEPENDENCIES = {
        vocalist_gender: { field: "source_type", equals: "vocal" },
        instrument_type: { field: "source_type", equals: "instrumental" },
        skip_separation: { field: "source_type", equals: "instrumental" }
    };

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
            dropzone.textContent = "Uploaded: " + payload.filename;
            setStatus("Audio uploaded and autofilled: " + payload.filename);
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
                    attachAudioDropzone(row, input);
                }

                section.appendChild(row);
            });

            schemaForms.appendChild(section);
        });

        bindConditionalVisibilityHandlers();
        updateConditionalVisibility();
    }

    function bindConditionalVisibilityHandlers() {
        const sourceInput = document.getElementById(fieldInputId("source_type"));
        if (!sourceInput) return;
        sourceInput.addEventListener("change", function () {
            updateConditionalVisibility();
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
    }

    function renderArtifacts(job) {
        clearChildren(artifactListEl);
        const artifacts = job.artifacts || [];
        let preferredReport = null;

        artifacts.forEach(function (artifact) {
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

            if (!preferredReport && artifact.name.endsWith("report.html") && artifact.url) {
                preferredReport = artifact.url;
            }
        });

        if (preferredReport) {
            reportFrame.src = preferredReport;
        }
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
        renderArtifacts(job);

        if (job.status === "completed" && autoPrefilledFromJobId !== job.job_id) {
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

    async function submit(payload) {
        const res = await fetch("/api/jobs", {
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
            const job = await submit(payload);
            lastPayload = payload;
            rerunBtn.disabled = false;
            activeJobId = job.job_id;
            setBusy(true);
            setStatus("Submitted: " + job.job_id);
            logsEl.textContent = "";
            artifactListEl.textContent = "";
            reportFrame.src = "";
            startPolling(job.job_id);
            await refreshJob(job.job_id);
        } catch (err) {
            setStatus("Error: " + err.message);
        }
    });

    rerunBtn.addEventListener("click", async function () {
        if (!lastPayload) return;
        try {
            const job = await submit(lastPayload);
            activeJobId = job.job_id;
            setBusy(true);
            setStatus("Rerun submitted: " + job.job_id);
            logsEl.textContent = "";
            artifactListEl.textContent = "";
            reportFrame.src = "";
            startPolling(job.job_id);
            await refreshJob(job.job_id);
        } catch (err) {
            setStatus("Error: " + err.message);
        }
    });

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
