(() => {
  const $ = (id) => document.getElementById(id);
  let data = null;
  let group = "";
  let working = false;
  let phraseSignature = "";
  let takeSignature = "";
  let referenceId = "";
  const selectedPhrases = new Set();
  const selectedTakes = new Set();

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function button(text, onClick, className = "secondary-button") {
    const node = element("button", text, className);
    node.type = "button";
    node.addEventListener("click", () => act(onClick));
    return node;
  }
  function notice(text, error = false) {
    $("notice").textContent = text;
    $("notice").dataset.error = String(error);
  }
  async function api(action, payload, binary = false) {
    const options = payload === undefined ? {} : {
      method: "POST",
      headers: {"X-Voice-IO": "response-studio", "Content-Type": binary ? "audio/wav" : "application/json"},
      body: binary ? payload : JSON.stringify(payload),
    };
    const response = await fetch(`/api/responses/${action}`, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
    if (result.state) update(result.state);
    return result;
  }
  async function act(fn) {
    if (working) return;
    working = true;
    toolbar();
    try { await fn(); }
    catch (error) { notice(error.message, true); }
    finally { working = false; toolbar(); }
  }
  function settings() {
    const result = {};
    for (const name of ["speed", "guidance_scale", "inference_steps", "duration"]) {
      const field = $(name);
      if (!field.checkValidity()) { field.reportValidity(); throw new Error(`Check ${name.replaceAll("_", " ")}.`); }
      result[name] = name === "duration" && !field.value ? null : Number(field.value);
    }
    for (const name of ["language", "reference_text", "instruct"]) {
      result[name] = $(name === "reference_text" ? "reference-text" : name).value;
    }
    for (const name of ["denoise", "preprocess_prompt", "postprocess_output"]) result[name] = $(name).checked;
    return result;
  }
  function fillSettings(saved) {
    for (const [name, value] of Object.entries(saved)) {
      const field = $(name === "reference_text" ? "reference-text" : name);
      if (typeof value === "boolean") field.checked = value;
      else field.value = value === null ? "" : value;
    }
  }
  function currentPhrases() {
    return data.phrases.filter(p => p.group === group && (!p.archived || $("show-archived").checked));
  }
  function currentTakes() {
    const filter = $("filter").value;
    return data.candidates.filter(c => c.group === group && (filter === "all" || (filter === "published" ? c.published : c.status === filter)));
  }
  function kept() { return data.candidates.filter(c => c.group === group && c.status === "kept" && !c.published); }
  function selection() { return data.candidates.filter(c => selectedTakes.has(c.id)); }
  function toolbar() {
    if (!data) return;
    const count = selectedPhrases.size * Number($("takes").value);
    const canGenerate = !working && !data.busy && Boolean(data.reference) && $("consent").checked;
    $("generate").disabled = !canGenerate || !Number.isInteger(count) || count < 1 || count > 64 || !$("takes").checkValidity();
    $("generation-summary").textContent = selectedPhrases.size ? `${selectedPhrases.size} phrase(s) × ${$("takes").value} takes = ${count} candidates${count > 64 ? " — reduce to 64 or fewer" : ""}.` : "Select phrases to generate.";
    const chosen = selection();
    const reviewable = chosen.length > 0 && chosen.length <= 64 && chosen.every(c => !c.published);
    $("keep-selected").disabled = working || !reviewable;
    $("reject-selected").disabled = working || !reviewable;
    $("regenerate-selected").disabled = !canGenerate || !reviewable || chosen.some(c => c.status !== "rejected" || c.replacement_id);
    const ready = kept();
    $("publish-count").textContent = ready.length ? `${ready.length} kept take(s) ready for playback` : "No kept takes to publish";
    $("publish").textContent = ready.length > 64 ? "Publish next 64 kept takes" : "Publish kept takes";
    $("publish").disabled = working || ready.length === 0;
    $("upload-reference").disabled = working || !$("reference-file").files.length;
    $("save-settings").disabled = working;
    $("retry-job").disabled = !canGenerate;
    document.querySelectorAll("[data-regenerate]").forEach(b => { b.disabled = !canGenerate; });
  }
  function renderPhrases() {
    const container = $("phrases");
    container.replaceChildren();
    const phrases = currentPhrases();
    if (!phrases.length) container.append(element("p", "No phrases here yet. Add your first reply below.", "muted"));
    for (const phrase of phrases) {
      const row = element("div", undefined, `phrase-row${phrase.archived ? " archived" : ""}`);
      const label = element("label", undefined, "check");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = selectedPhrases.has(phrase.id);
      checkbox.disabled = phrase.archived;
      checkbox.setAttribute("aria-label", `Select phrase: ${phrase.text}`);
      checkbox.addEventListener("change", () => { checkbox.checked ? selectedPhrases.add(phrase.id) : selectedPhrases.delete(phrase.id); toolbar(); });
      label.append(checkbox, element("span", phrase.text));
      const actions = element("div", undefined, "phrase-actions");
      if (!phrase.archived) actions.append(button("Edit", () => {
        if (row.querySelector(".phrase-editor")) return;
        const editor = element("form", undefined, "phrase-editor");
        const input = document.createElement("textarea");
        input.value = phrase.text;
        input.maxLength = 300;
        input.required = true;
        input.setAttribute("aria-label", "Edit reply phrase");
        const controls = element("div");
        const save = element("button", "Save phrase", "secondary-button");
        save.type = "submit";
        controls.append(save, button("Cancel", () => editor.remove(), "text-button"));
        editor.append(input, controls);
        editor.addEventListener("submit", e => { e.preventDefault(); act(async () => {
          await api("phrases", {id: phrase.id, group, text: input.value});
          editor.remove();
          notice("Phrase saved. Existing takes keep their original text.");
        }); });
        row.append(editor);
        input.focus();
      }, "text-button"));
      actions.append(button(phrase.archived ? "Restore" : "Archive", async () => {
        selectedPhrases.delete(phrase.id);
        await api("archive", {id: phrase.id, archived: !phrase.archived});
        notice(phrase.archived ? "Phrase restored." : "Phrase archived. Existing takes are unchanged.");
      }, "text-button"));
      row.append(label, actions);
      container.append(row);
    }
  }
  async function review(ids, status) {
    selectedTakes.clear();
    await api("review", {candidate_ids: ids, status});
    notice(status === "kept" ? "Takes kept. Publish them when you are ready." : "Takes rejected and retained for comparison.");
  }
  async function generate(payload) {
    const request = {request_id: crypto.randomUUID().replaceAll("-", ""), settings: settings(), batch_size: Number($("batch-size").value), consent_to_upload: $("consent").checked, ...payload};
    const result = await api("generate", request);
    notice(`Queued ${result.job.total} take(s). You can leave this page open or return later.`);
    await refresh();
  }
  function renderTakes() {
    const container = $("candidates");
    container.replaceChildren();
    const takes = currentTakes();
    const all = data.candidates.filter(c => c.group === group);
    $("take-count").textContent = `${all.length} total · ${all.filter(c => c.status === "pending").length} awaiting review · ${all.filter(c => c.published).length} published`;
    if (!takes.length) {
      const empty = element("div", undefined, "empty-state");
      empty.append(element("h3", all.length ? "No takes match this filter" : "Your first takes will appear here"), element("p", all.length ? "Choose another filter to see saved takes." : "Save a reference, select reply phrases, and generate a few takes to compare."));
      container.append(empty);
      return;
    }
    const groups = new Map();
    for (const take of takes) {
      const key = `${take.phrase_id}:${take.text}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(take);
    }
    for (const rows of groups.values()) {
      const section = element("section", undefined, "take-group");
      section.append(element("h3", rows[0].text));
      for (const take of rows) {
        const index = data.candidates.filter(c => c.phrase_id === take.phrase_id).findIndex(c => c.id === take.id) + 1;
        const row = element("article", undefined, "take-row");
        row.dataset.status = take.status;
        const header = element("div", undefined, "take-header");
        const label = element("label", undefined, "check take-label");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = selectedTakes.has(take.id);
        checkbox.setAttribute("aria-label", `Select take ${index}: ${take.text}`);
        checkbox.addEventListener("change", () => { checkbox.checked ? selectedTakes.add(take.id) : selectedTakes.delete(take.id); toolbar(); });
        label.append(checkbox, element("span", `Take ${index} · ${take.seconds.toFixed(1)} s`));
        const status = take.published ? "Published" : take.replacement_id && take.status === "rejected" ? "Rejected · replacement generated" : {pending: "Awaiting review", kept: "Kept", rejected: "Rejected"}[take.status];
        header.append(label, element("span", status, "take-status"));
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = take.audio_url;
        audio.setAttribute("aria-label", `Preview take ${index}: ${take.text}`);
        const actions = element("div", undefined, "take-actions");
        if (take.published) {
          actions.append(button("Remove from playback", async () => {
            const result = await api("unpublish", {id: take.id});
            publicationNotice(result, "Removed from playback. The kept take is still saved.");
          }));
        } else {
          const keep = button("Keep", () => review([take.id], "kept"));
          const reject = button("Reject", () => review([take.id], "rejected"));
          keep.disabled = take.status === "kept";
          reject.disabled = take.status === "rejected";
          keep.setAttribute("aria-label", `Keep take ${index}: ${take.text}`);
          reject.setAttribute("aria-label", `Reject take ${index}: ${take.text}`);
          actions.append(keep, reject);
          if (take.status === "rejected" && !take.replacement_id) {
            const regenerate = button("Regenerate", () => generate({candidate_ids: [take.id]}));
            regenerate.dataset.regenerate = "true";
            actions.append(regenerate);
          }
        }
        const details = document.createElement("details");
        details.append(element("summary", `Speed ${take.settings.speed} · CFG ${take.settings.guidance_scale} · ${take.settings.inference_steps} steps`), element("pre", JSON.stringify({created_at: take.created_at, ...take.settings}, null, 2)));
        row.append(header, audio, actions, details);
        section.append(row);
      }
      container.append(section);
    }
  }
  function renderJob() {
    const job = data.jobs.at(-1);
    $("job-panel").hidden = !job;
    if (!job) return;
    const active = ["queued", "running"].includes(job.status);
    const status = job.status === "completed" ? "Ready to review" : job.status;
    $("job-status").textContent = `${job.completed} of ${job.total} takes saved · ${status}${job.cancel_requested && active ? " · stopping after this batch" : ""}`;
    $("job-progress").max = job.total;
    $("job-progress").value = job.completed;
    $("job-error").textContent = job.error;
    $("cancel-job").hidden = !active;
    $("cancel-job").disabled = Boolean(job.cancel_requested);
    $("retry-job").hidden = active || job.completed === job.total || Boolean(job.retried_by);
  }
  function update(next, initial = false) {
    data = next;
    if (initial) {
      $("group").replaceChildren(...data.groups.map(g => {
        const option = element("option", g.name.replaceAll("_", " "));
        option.value = g.name;
        return option;
      }));
      group = data.groups[0]?.name || "";
      fillSettings(data.settings);
    }
    const info = data.groups.find(g => g.name === group);
    $("group-context").textContent = info?.commands.length ? `Used by: ${info.commands.map(c => c.name).join(", ")}` : group === "start" ? "Wake-phrase acknowledgement" : "Event response";
    const newReference = data.reference?.id || "";
    if (newReference !== referenceId) {
      referenceId = newReference;
      $("reference-audio").hidden = !referenceId;
      if (referenceId) {
        $("reference-audio").src = `${data.reference.audio_url}?id=${referenceId}`;
        $("reference-info").textContent = `Saved locally · ${data.reference.seconds.toFixed(1)} s · ${(data.reference.sample_rate / 1000).toFixed(1)} kHz. Only sent to the Space when you generate.`;
      }
    }
    const ps = JSON.stringify([group, data.phrases, $("show-archived").checked]);
    if (ps !== phraseSignature) { phraseSignature = ps; renderPhrases(); }
    const ts = JSON.stringify([group, data.candidates, $("filter").value]);
    if (ts !== takeSignature) { takeSignature = ts; renderTakes(); }
    renderJob();
    toolbar();
  }
  async function refresh(initial = false) { update(await api("state"), initial || !data); }
  function publicationNotice(result, message) {
    if (result.listener_error) notice(`${message} Listener restart failed: ${result.listener_error}`, true);
    else notice(`${message} ${result.listener_updated ? "Listener refreshed." : "Restart your listener to load the change."}`);
  }

  $("refresh").addEventListener("click", () => act(() => refresh()));
  $("group").addEventListener("change", () => { group = $("group").value; selectedPhrases.clear(); selectedTakes.clear(); update(data); });
  $("show-archived").addEventListener("change", () => update(data));
  $("filter").addEventListener("change", () => { selectedTakes.clear(); update(data); });
  $("reference-file").addEventListener("change", toolbar);
  $("consent").addEventListener("change", toolbar);
  $("takes").addEventListener("input", toolbar);
  $("upload-reference").addEventListener("click", () => act(async () => {
    const file = $("reference-file").files[0];
    if (!file || file.size > 10 * 1024 * 1024) throw new Error("Choose a WAV file smaller than 10 MB.");
    await api("reference", file, true);
    $("reference-file").value = "";
    notice("Reference saved locally. Check its transcript before generating.");
  }));
  $("save-settings").addEventListener("click", () => act(async () => { await api("settings", {settings: settings()}); notice("Voice settings saved on this computer."); }));
  $("new-phrase-form").addEventListener("submit", e => { e.preventDefault(); act(async () => {
    await api("phrases", {group, text: $("new-phrase").value}); $("new-phrase").value = ""; notice("Reply phrase added.");
  }); });
  $("select-phrases").addEventListener("click", () => { const phrases = currentPhrases().filter(p => !p.archived); const clear = phrases.every(p => selectedPhrases.has(p.id)); phrases.forEach(p => clear ? selectedPhrases.delete(p.id) : selectedPhrases.add(p.id)); renderPhrases(); toolbar(); });
  $("select-takes").addEventListener("click", () => { const takes = currentTakes(); const clear = takes.every(c => selectedTakes.has(c.id)); takes.forEach(c => clear ? selectedTakes.delete(c.id) : selectedTakes.add(c.id)); renderTakes(); toolbar(); });
  $("generate").addEventListener("click", () => act(() => generate({phrase_ids: [...selectedPhrases], takes: Number($("takes").value)})));
  $("keep-selected").addEventListener("click", () => act(() => review([...selectedTakes], "kept")));
  $("reject-selected").addEventListener("click", () => act(() => review([...selectedTakes], "rejected")));
  $("regenerate-selected").addEventListener("click", () => act(() => generate({candidate_ids: [...selectedTakes]})));
  $("publish").addEventListener("click", () => act(async () => { const result = await api("publish", {candidate_ids: kept().slice(0, 64).map(c => c.id)}); publicationNotice(result, "Kept takes published to playback."); }));
  $("cancel-job").addEventListener("click", () => act(() => api("cancel", {job_id: data.jobs.at(-1).id})));
  $("retry-job").addEventListener("click", () => act(() => generate({retry_job_id: data.jobs.at(-1).id})));
  document.addEventListener("play", e => { if (e.target.tagName === "AUDIO") document.querySelectorAll("audio").forEach(a => { if (a !== e.target) a.pause(); }); }, true);
  refresh(true).then(() => notice("Workspace ready. Only published takes are used for playback.")).catch(e => notice(e.message, true));
  setInterval(() => { if (data?.busy && !working) refresh().catch(e => notice(`Could not refresh job: ${e.message}`, true)); }, 1500);
})();
