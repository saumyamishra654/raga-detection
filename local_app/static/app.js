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

    function createInput(field) {
        const id = fieldInputId(field.name);
        let input;

        if (field.choices && field.choices.length > 0) {
            input = document.createElement("select");
            field.choices.forEach(function (choice) {
                const option = document.createElement("option");
                option.value = String(choice);
                option.textContent = String(choice);
                if (field.default !== null && String(field.default) === String(choice)) {
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
            if (field.default !== null && field.default !== undefined) {
                input.value = String(field.default);
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

                section.appendChild(row);
            });

            schemaForms.appendChild(section);
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
            return parseInt(inputEl.value, 10);
        }
        if (field.value_type === "float") {
            return parseFloat(inputEl.value);
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

