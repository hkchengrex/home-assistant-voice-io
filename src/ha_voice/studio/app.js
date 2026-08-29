const elements = {
  commandList: document.querySelector("#command-list"),
  totalProgress: document.querySelector("#total-progress"),
  sampleLabel: document.querySelector("#sample-label"),
  phrase: document.querySelector("#phrase"),
  description: document.querySelector("#description"),
  sessionDot: document.querySelector("#session-dot"),
  sessionStatus: document.querySelector("#session-status"),
  listeningLane: document.querySelector("#listening-lane"),
  waveform: document.querySelector("#waveform"),
  recordButton: document.querySelector("#record-button"),
  recordLabel: document.querySelector("#record-label"),
  batchButton: document.querySelector("#batch-button"),
  levelFill: document.querySelector("#level-fill"),
  levelValue: document.querySelector("#level-value"),
  reviewPanel: document.querySelector("#review-panel"),
  playback: document.querySelector("#playback"),
  retryButton: document.querySelector("#retry-button"),
  keepButton: document.querySelector("#keep-button"),
  deviceSelect: document.querySelector("#device-select"),
  refreshButton: document.querySelector("#enable-button"),
  micHelp: document.querySelector("#mic-help"),
  message: document.querySelector("#message"),
  testButton: document.querySelector("#test-button"),
  savedSummary: document.querySelector("#saved-summary"),
  savedSelect: document.querySelector("#saved-select"),
  savedPlayback: document.querySelector("#saved-playback"),
  replaceTakeButton: document.querySelector("#replace-take-button"),
  continuousButton: document.querySelector("#continuous-button"),
  continuousLabel: document.querySelector("#continuous-label"),
  captureWindow: document.querySelector("#capture-window"),
  savedTakes: document.querySelector("#saved-takes"),
  continuousHint: document.querySelector("#continuous-hint"),
  addCommandButton: document.querySelector("#add-command-button"),
  deleteCommandButton: document.querySelector("#delete-command-button"),
  deleteTakeButton: document.querySelector("#delete-take-button"),
  newCommandDialog: document.querySelector("#new-command-dialog"),
  newCommandForm: document.querySelector("#new-command-form"),
  newCommandUtterance: document.querySelector("#new-command-utterance"),
  newCommandDescription: document.querySelector("#new-command-description"),
  newCommandTarget: document.querySelector("#new-command-target"),
  cancelCommandButton: document.querySelector("#cancel-command-button"),
  compareMicsButton: document.querySelector("#compare-mics-button"),
  micCompareDialog: document.querySelector("#mic-compare-dialog"),
  compareSetup: document.querySelector("#compare-setup"),
  compareRun: document.querySelector("#compare-run"),
  compareDeviceA: document.querySelector("#compare-device-a"),
  compareDeviceB: document.querySelector("#compare-device-b"),
  comparePhrase: document.querySelector("#compare-phrase"),
  comparePairCount: document.querySelector("#compare-pair-count"),
  closeCompareButton: document.querySelector("#close-compare-button"),
  cancelCompareButton: document.querySelector("#cancel-compare-button"),
  startCompareButton: document.querySelector("#start-compare-button"),
  recordCompareButton: document.querySelector("#record-compare-button"),
  recordCompareLabel: document.querySelector("#record-compare-label"),
  compareProgress: document.querySelector("#compare-progress"),
  comparePromptText: document.querySelector("#compare-prompt-text"),
  compareNameA: document.querySelector("#compare-name-a"),
  compareNameB: document.querySelector("#compare-name-b"),
  compareSummaryA: document.querySelector("#compare-summary-a"),
  compareSummaryB: document.querySelector("#compare-summary-b"),
  compareTakesA: document.querySelector("#compare-takes-a"),
  compareTakesB: document.querySelector("#compare-takes-b"),
  compareVerdict: document.querySelector("#compare-verdict"),
  restartCompareButton: document.querySelector("#restart-compare-button"),
  finishCompareButton: document.querySelector("#finish-compare-button"),
  batchDialog: document.querySelector("#batch-dialog"),
  batchSetup: document.querySelector("#batch-setup"),
  batchRun: document.querySelector("#batch-run"),
  batchSetupPhrase: document.querySelector("#batch-setup-phrase"),
  batchDeviceName: document.querySelector("#batch-device-name"),
  batchCount: document.querySelector("#batch-count"),
  cancelBatchButton: document.querySelector("#cancel-batch-button"),
  startBatchButton: document.querySelector("#start-batch-button"),
  batchProgress: document.querySelector("#batch-progress"),
  batchSavedCount: document.querySelector("#batch-saved-count"),
  batchCue: document.querySelector("#batch-cue"),
  batchStatus: document.querySelector("#batch-status"),
  batchPrompt: document.querySelector("#batch-prompt"),
  batchProgressFill: document.querySelector("#batch-progress-fill"),
  batchDetail: document.querySelector("#batch-detail"),
  retryBatchButton: document.querySelector("#retry-batch-button"),
  stopBatchButton: document.querySelector("#stop-batch-button"),
  doneBatchButton: document.querySelector("#done-batch-button"),
  triggerReviewSummary: document.querySelector("#trigger-review-summary"),
  triggerReviewList: document.querySelector("#trigger-review-list"),
  refreshTriggersButton: document.querySelector("#refresh-triggers-button"),
};

const state = {
  commands: [],
  devices: [],
  activeIndex: 0,
  recording: false,
  reviewBlob: null,
  reviewUrl: null,
  testMode: false,
  continuousRunning: false,
  continuousEventId: 0,
  continuousNoticeUntil: 0,
  continuousAwaitingCommand: false,
  continuousCommandDeadline: 0,
  commandTimeoutSeconds: 3,
  startPhraseReady: false,
  startPhraseEnabled: false,
  replaceFilename: null,
  replaceTakeLabel: null,
  canManageCommands: false,
  batchRunning: false,
  batchBusy: false,
  batchStopRequested: false,
  batchTarget: 0,
  batchCompleted: 0,
  batchRunId: 0,
  batchError: "",
  diagnosticsLoading: false,
  diagnosticSignature: "",
};

const comparison = {
  running: false,
  busy: false,
  channel: 0,
  pairIndex: 0,
  pairTarget: 3,
  phrase: "Start phrase",
  devices: [],
  takes: [[], []],
  runId: 0,
  error: "",
};

function activeCommand() {
  return state.commands[state.activeIndex] || null;
}

function setMessage(message, kind = "normal") {
  elements.message.textContent = message;
  elements.message.dataset.kind = kind;
}

function setSession(label, mode = "idle") {
  elements.sessionStatus.textContent = label;
  elements.sessionDot.className = `session-dot${mode === "ready" ? " is-ready" : mode === "live" ? " is-live" : mode === "warning" ? " is-warning" : ""}`;
}

function batchUnavailable() {
  return state.devices.length === 0
    || state.recording
    || Boolean(state.reviewBlob)
    || state.testMode
    || state.continuousRunning
    || state.batchRunning;
}

function renderBatchAvailability() {
  elements.batchButton.disabled = batchUnavailable();
}

function setBatchInteractionLocked(locked) {
  elements.deviceSelect.disabled = locked || state.devices.length === 0 || state.continuousRunning;
  elements.refreshButton.disabled = locked || state.continuousRunning;
  elements.recordButton.disabled = locked || state.devices.length === 0 || state.continuousRunning;
  elements.compareMicsButton.disabled = locked || comparableDevices().length < 2 || state.continuousRunning;
  elements.testButton.disabled = locked || state.continuousRunning;
  elements.continuousButton.disabled = locked;
  elements.addCommandButton.disabled = locked;
  const hasSavedTake = Boolean(elements.savedSelect.selectedOptions[0]?.dataset.filename);
  elements.replaceTakeButton.disabled = locked || state.continuousRunning || !hasSavedTake;
  elements.deleteTakeButton.disabled = locked || state.continuousRunning || !hasSavedTake;
  renderBatchAvailability();
}

function renderCommands() {
  elements.commandList.replaceChildren();
  let complete = 0;
  let target = 0;
  state.commands.forEach((command, index) => {
    complete += Math.min(command.count, command.target_samples);
    target += command.target_samples;
    const button = document.createElement("button");
    button.type = "button";
    button.className = `command-button${command.count >= command.target_samples ? " is-complete" : ""}`;
    button.setAttribute("aria-current", index === state.activeIndex ? "true" : "false");
    button.disabled = state.batchRunning;
    if (state.testMode) button.setAttribute("aria-current", "false");
    button.innerHTML = `
      <span>
        <span class="command-name"></span>
        <span class="command-description"></span>
      </span>
      <span class="command-count"></span>
    `;
    button.querySelector(".command-name").textContent = command.utterance;
    button.querySelector(".command-description").textContent = command.description;
    button.querySelector(".command-count").textContent = `${command.count}/${command.target_samples}`;
    button.addEventListener("click", () => selectCommand(index));
    elements.commandList.append(button);
  });
  elements.testButton.setAttribute(
    "aria-pressed",
    state.testMode && !state.continuousRunning ? "true" : "false",
  );
  elements.totalProgress.textContent = `${complete} of ${target} samples`;
  renderBatchAvailability();
}

function renderActiveCommand() {
  const selected = activeCommand();
  const regularCommands = state.commands.filter(
    (command) => !command.is_negative && !command.is_start_phrase,
  );
  elements.deleteCommandButton.hidden = !(
    state.canManageCommands
    && selected
    && !selected.is_negative
    && !selected.is_start_phrase
    && regularCommands.length > 1
    && !state.testMode
    && !state.continuousRunning
  );
  if (state.continuousRunning) {
    elements.sampleLabel.textContent = "Continuous test · no action will run";
    elements.phrase.textContent = state.startPhraseReady
      ? "Waiting for start phrase"
      : "Listening for a command";
    elements.description.textContent = state.startPhraseReady
      ? `Say the start phrase, then say one command within ${state.commandTimeoutSeconds} seconds.`
      : "Speak naturally, then pause briefly for recognition.";
    elements.recordLabel.textContent = "Listening continuously";
    return;
  }
  if (state.testMode) {
    elements.sampleLabel.textContent = "Live test · no action will run";
    elements.phrase.textContent = "Say any command";
    elements.description.textContent = "The matcher will identify it or safely reject it.";
    elements.recordLabel.textContent = "Test a command";
    return;
  }
  const command = activeCommand();
  if (!command) return;
  const nextSample = Math.min(command.count + 1, command.target_samples);
  elements.sampleLabel.textContent = command.count >= command.target_samples
    ? `Set complete · ${command.count} samples`
    : `Sample ${nextSample} of ${command.target_samples}`;
  elements.phrase.textContent = command.is_negative || command.is_start_phrase
    ? command.utterance
    : `“${command.utterance}”`;
  elements.description.textContent = command.description;
}

function renderDevices(selectedId = "") {
  elements.deviceSelect.replaceChildren();
  state.devices.forEach((device) => {
    const option = document.createElement("option");
    option.value = String(device.id);
    option.textContent = `${device.name}${device.channels > 1 ? ` · ${device.channels} ch` : ""}`;
    elements.deviceSelect.append(option);
  });
  const rememberedDevice = selectedId || window.localStorage.getItem("voiceprint-input-device") || "";
  if (rememberedDevice && state.devices.some((device) => String(device.id) === rememberedDevice)) {
    elements.deviceSelect.value = rememberedDevice;
  } else {
    const preferred = state.devices.find((device) => device.name.toLowerCase() === "default")
      || state.devices.find((device) => device.name.toLowerCase().includes("usb pnp"))
      || state.devices.find((device) => device.name.toLowerCase().includes("microphone array"));
    if (preferred) elements.deviceSelect.value = String(preferred.id);
  }
  const ready = state.devices.length > 0;
  elements.deviceSelect.disabled = !ready;
  elements.recordButton.disabled = !ready || state.continuousRunning;
  renderBatchAvailability();
  elements.compareMicsButton.disabled = comparableDevices().length < 2 || state.continuousRunning;
  elements.recordLabel.textContent = ready ? "Record sample" : "No microphone found";
  if (ready) {
    elements.micHelp.textContent = "Captured by the local studio service—no browser permission required.";
    setSession("Ready", "ready");
    setMessage("Choose a microphone, then record a natural speaking voice.");
  } else {
    elements.deviceSelect.append(new Option("No microphone found", ""));
    elements.micHelp.textContent = "This computer did not report an available input device.";
    setSession("No microphone");
    setMessage("Connect or enable a microphone, then refresh devices.", "error");
  }
}

function comparisonDeviceName(channel) {
  return comparison.devices[channel]?.name || `Microphone ${channel === 0 ? "A" : "B"}`;
}

function comparableDevices() {
  return state.devices.filter((device) => device.comparison_eligible !== false);
}

function populateComparisonDevices() {
  const devices = comparableDevices();
  [elements.compareDeviceA, elements.compareDeviceB].forEach((select) => {
    select.replaceChildren();
    devices.forEach((device) => {
      select.append(new Option(device.name, String(device.id)));
    });
  });
  const usb = devices.find((device) => device.name.toLowerCase().includes("usb"));
  const other = devices.find((device) => (
    device.id !== usb?.id
  )) || devices.find((device) => device.id !== usb?.id);
  if (other && usb) {
    elements.compareDeviceA.value = String(other.id);
    elements.compareDeviceB.value = String(usb.id);
  } else if (devices.length >= 2) {
    elements.compareDeviceA.value = String(devices[0].id);
    elements.compareDeviceB.value = String(devices[1].id);
  }
}

function clearComparisonRecordings() {
  comparison.runId += 1;
  comparison.takes.flat().forEach((take) => URL.revokeObjectURL(take.url));
  comparison.takes = [[], []];
  comparison.running = false;
  comparison.busy = false;
  comparison.channel = 0;
  comparison.pairIndex = 0;
  comparison.error = "";
}

function openMicrophoneComparison() {
  if (state.recording || state.reviewBlob || state.continuousRunning) {
    setMessage("Finish the current recording or continuous test before comparing microphones.", "error");
    return;
  }
  if (comparableDevices().length < 2) {
    setMessage("Connect two microphones, then refresh devices.", "error");
    return;
  }
  clearComparisonRecordings();
  populateComparisonDevices();
  elements.compareSetup.hidden = false;
  elements.compareRun.hidden = true;
  elements.micCompareDialog.showModal();
  elements.compareDeviceA.focus();
}

function closeMicrophoneComparison() {
  if (elements.micCompareDialog.open) elements.micCompareDialog.close();
}

function percentile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

async function measureComparisonAudio(blob) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const audioContext = new AudioContextClass();
  try {
    const decoded = await audioContext.decodeAudioData(await blob.arrayBuffer());
    const samples = decoded.getChannelData(0);
    const frameLength = Math.max(1, Math.round(decoded.sampleRate * 0.02));
    const frameRms = [];
    let totalSquares = 0;
    let peak = 0;
    let clipped = 0;
    for (let start = 0; start < samples.length; start += frameLength) {
      let frameSquares = 0;
      const end = Math.min(samples.length, start + frameLength);
      for (let index = start; index < end; index += 1) {
        const magnitude = Math.abs(samples[index]);
        const square = samples[index] * samples[index];
        frameSquares += square;
        totalSquares += square;
        peak = Math.max(peak, magnitude);
        if (magnitude >= 0.999) clipped += 1;
      }
      frameRms.push(Math.sqrt(frameSquares / Math.max(1, end - start)));
    }
    const dbfs = (value) => 20 * Math.log10(Math.max(value, 0.000001));
    const noise = percentile(frameRms, 0.2);
    const speech = percentile(frameRms, 0.9);
    return {
      level: dbfs(Math.sqrt(totalSquares / Math.max(1, samples.length))),
      noise: dbfs(noise),
      speech: dbfs(speech),
      spread: Math.max(0, dbfs(speech) - dbfs(noise)),
      peak: dbfs(peak),
      clipping: clipped / Math.max(1, samples.length) * 100,
    };
  } finally {
    await audioContext.close();
  }
}

function averageMetric(takes, key) {
  return takes.reduce((sum, take) => sum + take.metrics[key], 0) / Math.max(1, takes.length);
}

function renderComparisonTakes(channel) {
  const takes = comparison.takes[channel];
  const summary = channel === 0 ? elements.compareSummaryA : elements.compareSummaryB;
  const container = channel === 0 ? elements.compareTakesA : elements.compareTakesB;
  container.replaceChildren();
  if (!takes.length) {
    summary.textContent = "No takes yet";
    return;
  }
  const voice = averageMetric(takes, "speech");
  const noise = averageMetric(takes, "noise");
  const spread = averageMetric(takes, "spread");
  const clipping = averageMetric(takes, "clipping");
  summary.innerHTML = `<strong>${takes.length} take${takes.length === 1 ? "" : "s"}</strong><br>Voice ${voice.toFixed(1)} dB · room ${noise.toFixed(1)} dB · spread ${spread.toFixed(1)} dB${clipping >= 0.05 ? " · clipping detected" : ""}`;
  takes.forEach((take, index) => {
    const row = document.createElement("div");
    row.className = "compare-take";
    const label = document.createElement("span");
    label.textContent = `Pair ${index + 1}`;
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = take.url;
    row.append(label, audio);
    container.append(row);
  });
}

function renderComparisonVerdict() {
  const first = comparison.takes[0];
  const second = comparison.takes[1];
  if (comparison.error) {
    elements.compareVerdict.textContent = comparison.error;
    return;
  }
  if (!first.length || !second.length) {
    elements.compareVerdict.textContent = "Record both sides of the pair, then listen back.";
    return;
  }
  if (comparison.running) {
    elements.compareVerdict.innerHTML = first.length === second.length
      ? `<strong>Pair ${first.length} captured.</strong> Replay A and B above; the test will keep alternating under the same conditions.`
      : "Now record the same phrase through microphone B without changing position.";
    return;
  }
  const spreadA = averageMetric(first, "spread");
  const spreadB = averageMetric(second, "spread");
  const difference = Math.abs(spreadA - spreadB);
  if (difference < 2) {
    elements.compareVerdict.innerHTML = "<strong>The measurements are close.</strong> Replay each pair and choose the microphone with clearer words and less room sound.";
    return;
  }
  const cleaner = spreadA > spreadB ? "A" : "B";
  elements.compareVerdict.innerHTML = `<strong>Microphone ${cleaner} separated your voice from the room by ${difference.toFixed(1)} dB more.</strong> Confirm that advantage by replaying the paired clips.`;
}

function renderComparison() {
  elements.compareNameA.textContent = comparisonDeviceName(0);
  elements.compareNameB.textContent = comparisonDeviceName(1);
  elements.comparePromptText.textContent = comparison.phrase;
  renderComparisonTakes(0);
  renderComparisonTakes(1);
  renderComparisonVerdict();
  elements.recordCompareButton.classList.toggle("is-recording", comparison.busy);

  if (!comparison.running) {
    elements.compareProgress.textContent = "Paired test complete";
    elements.recordCompareLabel.textContent = "All pairs recorded";
    elements.recordCompareButton.disabled = true;
    return;
  }
  const letter = comparison.channel === 0 ? "A" : "B";
  elements.compareProgress.textContent = `Pair ${comparison.pairIndex + 1} of ${comparison.pairTarget} · microphone ${letter}`;
  elements.recordCompareLabel.textContent = comparison.busy
    ? `Recording microphone ${letter}…`
    : `Record microphone ${letter}`;
  elements.recordCompareButton.disabled = comparison.busy;
}

function startMicrophoneComparison() {
  const firstId = elements.compareDeviceA.value;
  const secondId = elements.compareDeviceB.value;
  const phrase = elements.comparePhrase.value.trim();
  if (!firstId || !secondId || firstId === secondId) {
    elements.compareDeviceB.setCustomValidity("Choose a different microphone.");
    elements.compareDeviceB.reportValidity();
    elements.compareDeviceB.focus();
    return;
  }
  elements.compareDeviceB.setCustomValidity("");
  if (!phrase) {
    elements.comparePhrase.setCustomValidity("Enter one phrase to repeat.");
    elements.comparePhrase.reportValidity();
    elements.comparePhrase.focus();
    return;
  }
  elements.comparePhrase.setCustomValidity("");
  clearComparisonRecordings();
  comparison.devices = [firstId, secondId].map((id) => state.devices.find((device) => String(device.id) === id));
  comparison.pairTarget = Number(elements.comparePairCount.value) || 3;
  comparison.phrase = phrase;
  comparison.running = true;
  elements.compareSetup.hidden = true;
  elements.compareRun.hidden = false;
  renderComparison();
  elements.recordCompareButton.focus();
}

async function recordComparisonTake() {
  if (!comparison.running || comparison.busy) return;
  comparison.busy = true;
  const channel = comparison.channel;
  const pairIndex = comparison.pairIndex;
  const runId = comparison.runId;
  const controller = new AbortController();
  const captureTimeout = window.setTimeout(() => controller.abort(), 8000);
  comparison.error = "";
  renderComparison();
  try {
    const response = await fetch("/api/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        device: Number(comparison.devices[channel].id),
        preserve_silence: true,
      }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || "Microphone capture failed");
    }
    const blob = await response.blob();
    const metrics = await measureComparisonAudio(blob);
    const url = URL.createObjectURL(blob);
    if (runId !== comparison.runId) {
      URL.revokeObjectURL(url);
      return;
    }
    comparison.takes[channel][pairIndex] = { url, metrics };
    if (channel === 0) {
      comparison.channel = 1;
    } else if (pairIndex + 1 < comparison.pairTarget) {
      comparison.channel = 0;
      comparison.pairIndex += 1;
    } else {
      comparison.running = false;
    }
  } catch (error) {
    comparison.error = error.name === "AbortError"
      ? "This microphone did not finish recording within eight seconds. Try this side again."
      : `${error.message} Try this side again.`;
  } finally {
    window.clearTimeout(captureTimeout);
    if (runId === comparison.runId) {
      comparison.busy = false;
      renderComparison();
      if (comparison.running) elements.recordCompareButton.focus();
    }
  }
}

function restartMicrophoneComparison() {
  clearComparisonRecordings();
  elements.compareSetup.hidden = false;
  elements.compareRun.hidden = true;
  elements.startCompareButton.focus();
}

function selectedDeviceName() {
  const id = elements.deviceSelect.value;
  return state.devices.find((device) => String(device.id) === id)?.name || "Selected microphone";
}

function openBatchRecorder() {
  const command = activeCommand();
  if (!command || batchUnavailable()) return;
  const remaining = Math.max(1, command.target_samples - command.count);
  state.batchTarget = Math.min(50, remaining);
  state.batchCompleted = 0;
  state.batchError = "";
  state.batchStopRequested = false;
  elements.batchCount.value = String(state.batchTarget);
  elements.batchSetupPhrase.textContent = command.utterance;
  elements.batchDeviceName.textContent = selectedDeviceName();
  elements.batchSetup.hidden = false;
  elements.batchRun.hidden = true;
  elements.batchDialog.showModal();
  elements.batchCount.focus();
  elements.batchCount.select();
}

function renderBatchRecorder() {
  const command = activeCommand();
  const next = Math.min(state.batchCompleted + 1, state.batchTarget);
  const finished = state.batchCompleted >= state.batchTarget;
  const stopping = state.batchStopRequested && state.batchBusy;
  elements.batchProgress.textContent = finished
    ? "Recording set complete"
    : `Take ${next} of ${state.batchTarget}`;
  elements.batchSavedCount.textContent = `${state.batchCompleted} saved`;
  elements.batchPrompt.textContent = command?.utterance || "Command";
  elements.batchProgressFill.style.width = `${state.batchTarget ? state.batchCompleted / state.batchTarget * 100 : 0}%`;
  elements.batchCue.dataset.state = state.batchError
    ? "error"
    : finished
      ? "complete"
      : state.batchBusy
        ? "recording"
        : "ready";

  if (state.batchError) {
    elements.batchStatus.textContent = "This take was not saved";
    elements.batchDetail.textContent = state.batchError;
  } else if (finished) {
    elements.batchStatus.textContent = "All recordings saved";
    elements.batchDetail.textContent = `${state.batchCompleted} new takes are now in the training library.`;
  } else if (state.batchBusy) {
    elements.batchStatus.textContent = "Chime, then speak now";
    elements.batchDetail.textContent = stopping
      ? "This take will save, then the recording set will stop."
      : "Recording for three seconds. The take saves automatically.";
  } else {
    elements.batchStatus.textContent = "Preparing the next cue";
    elements.batchDetail.textContent = "Every take is saved before the next one begins.";
  }

  elements.retryBatchButton.hidden = !state.batchError;
  elements.retryBatchButton.disabled = state.batchBusy;
  elements.stopBatchButton.hidden = finished;
  elements.stopBatchButton.disabled = stopping;
  elements.stopBatchButton.textContent = state.batchError
    ? "Finish set"
    : stopping
      ? "Stopping after this take…"
      : "Stop after this take";
  elements.doneBatchButton.hidden = !finished;
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function finishBatchRecorder({ stopped = false } = {}) {
  state.batchRunId += 1;
  state.batchRunning = false;
  state.batchBusy = false;
  state.batchStopRequested = false;
  let listenerError = "";
  let waitingForTemplates = false;
  try {
    const response = await fetch("/api/batch/finish", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not restore the listener");
    listenerError = payload.listener_error || "";
    waitingForTemplates = Boolean(payload.listener_waiting_for_templates);
  } catch (error) {
    listenerError = error.message;
  }

  await loadState({ preserveSelection: true }).catch(() => {});
  setBatchInteractionLocked(false);
  state.batchTarget = Math.max(state.batchTarget, state.batchCompleted);
  elements.batchSetup.hidden = true;
  elements.batchRun.hidden = false;
  elements.batchProgress.textContent = stopped ? "Recording set stopped" : "Recording set complete";
  elements.batchStatus.textContent = `${state.batchCompleted} recording${state.batchCompleted === 1 ? "" : "s"} saved`;
  elements.batchDetail.textContent = listenerError
    ? `The recordings are safe, but the listener could not reload: ${listenerError}`
    : waitingForTemplates
      ? "The listener will start after the first command recording is added."
      : "The matcher has reloaded the updated training library.";
  elements.batchCue.dataset.state = listenerError ? "error" : "complete";
  elements.batchProgressFill.style.width = "100%";
  elements.retryBatchButton.hidden = true;
  elements.stopBatchButton.hidden = true;
  elements.doneBatchButton.hidden = false;
  elements.doneBatchButton.focus();
  renderBatchAvailability();
  setSession("Ready", "ready");
  setMessage(listenerError
    ? `Recordings saved, but the listener could not reload: ${listenerError}`
    : `${state.batchCompleted} recordings saved.`,
  listenerError ? "error" : "success");
}

async function runBatchRecorder(runId) {
  while (
    runId === state.batchRunId
    && state.batchRunning
    && !state.batchStopRequested
    && state.batchCompleted < state.batchTarget
  ) {
    state.batchBusy = true;
    state.batchError = "";
    renderBatchRecorder();
    const command = activeCommand();
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch(`/api/batch/capture/${encodeURIComponent(command.name)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device: Number(elements.deviceSelect.value) }),
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "This take could not be recorded");
      command.count = payload.count;
      state.batchCompleted += 1;
      state.batchBusy = false;
      renderCommands();
      renderBatchRecorder();
      if (state.batchStopRequested || state.batchCompleted >= state.batchTarget) break;
      await delay(650);
    } catch (error) {
      state.batchBusy = false;
      state.batchError = error.name === "AbortError"
        ? "The microphone did not finish within ten seconds. Check it, then retry this take."
        : error.message;
      renderBatchRecorder();
      return;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  if (runId !== state.batchRunId || !state.batchRunning) return;
  await finishBatchRecorder({ stopped: state.batchStopRequested });
}

function startBatchRecorder() {
  const count = Number(elements.batchCount.value);
  if (!Number.isInteger(count) || count < 1 || count > 50) {
    elements.batchCount.setCustomValidity("Choose between 1 and 50 recordings.");
    elements.batchCount.reportValidity();
    return;
  }
  elements.batchCount.setCustomValidity("");
  state.batchTarget = count;
  state.batchCompleted = 0;
  state.batchError = "";
  state.batchStopRequested = false;
  state.batchBusy = false;
  state.batchRunning = true;
  state.batchRunId += 1;
  elements.batchSetup.hidden = true;
  elements.batchRun.hidden = false;
  setBatchInteractionLocked(true);
  renderCommands();
  renderBatchRecorder();
  setSession("Batch recording", "live");
  setMessage("Batch recording is active. Listen for each chime, then speak.");
  runBatchRecorder(state.batchRunId);
}

function stopBatchRecorder() {
  if (!state.batchRunning) return;
  if (state.batchError || !state.batchBusy) {
    finishBatchRecorder({ stopped: true });
    return;
  }
  state.batchStopRequested = true;
  renderBatchRecorder();
}

function retryBatchRecorder() {
  if (!state.batchRunning || !state.batchError || state.batchBusy) return;
  state.batchError = "";
  state.batchStopRequested = false;
  renderBatchRecorder();
  runBatchRecorder(state.batchRunId);
}

function closeBatchRecorder() {
  if (state.batchRunning) return;
  if (elements.batchDialog.open) elements.batchDialog.close();
}

async function loadState({ preserveSelection = false } = {}) {
  const activeName = preserveSelection ? activeCommand()?.name : "";
  const selectedDevice = elements.deviceSelect.value;
  const response = await fetch("/api/state", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load the training set");
  const payload = await response.json();
  state.commands = payload.commands;
  state.devices = payload.devices || [];
  state.startPhraseReady = Boolean(payload.start_phrase_ready);
  state.startPhraseEnabled = Boolean(payload.start_phrase_enabled);
  state.commandTimeoutSeconds = Number(payload.command_timeout_seconds) || 3;
  state.canManageCommands = Boolean(payload.can_manage_commands);
  elements.addCommandButton.hidden = !state.canManageCommands;
  elements.continuousHint.textContent = state.startPhraseReady
    ? `Start phrase → one command within ${state.commandTimeoutSeconds} seconds. Listening stops when this page closes.`
    : "Direct commands remain active until Start phrase is trained and enabled.";
  if (activeName) {
    const index = state.commands.findIndex((command) => command.name === activeName);
    state.activeIndex = index >= 0 ? index : 0;
  }
  renderCommands();
  renderActiveCommand();
  renderDevices(selectedDevice);
  await loadSavedTakes();
  drawIdleWave();
}

async function loadSavedTakes() {
  const command = activeCommand();
  if (!command) return;
  const requestedName = command.name;
  elements.savedSelect.disabled = true;
  elements.replaceTakeButton.disabled = true;
  elements.deleteTakeButton.disabled = true;
  elements.savedSelect.replaceChildren(new Option("Loading…", ""));
  elements.savedPlayback.pause();
  elements.savedPlayback.removeAttribute("src");
  elements.savedPlayback.hidden = true;
  elements.savedSummary.textContent = `Loading “${command.utterance}”…`;

  const response = await fetch(`/api/samples/${encodeURIComponent(requestedName)}`, {
    cache: "no-store",
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Could not load saved takes");
  if (activeCommand()?.name !== requestedName) return;

  elements.savedSelect.replaceChildren();
  if (!payload.samples.length) {
    elements.savedSelect.append(new Option("No saved takes", ""));
    elements.savedSummary.textContent = `No recordings saved for “${command.utterance}”.`;
    return;
  }
  payload.samples.forEach((sample, index) => {
    const option = new Option(`Take ${index + 1}`, sample.url);
    option.dataset.filename = sample.filename;
    elements.savedSelect.append(option);
  });
  elements.savedSelect.disabled = false;
  elements.replaceTakeButton.disabled = state.continuousRunning;
  elements.deleteTakeButton.disabled = state.continuousRunning;
  elements.savedPlayback.src = payload.samples[0].url;
  elements.savedPlayback.hidden = false;
  elements.savedSummary.textContent = `${payload.samples.length} local recording${payload.samples.length === 1 ? "" : "s"} · “${command.utterance}”`;
}

function diagnosticMetric(result) {
  if (!result) return "No command followed";
  const score = Number(result.score);
  const margin = Number(result.margin);
  const scoreText = Number.isFinite(score) ? score.toFixed(3) : "—";
  const marginText = Number.isFinite(margin) ? `${(margin * 100).toFixed(1)}%` : "—";
  return `score ${scoreText} · margin ${marginText}`;
}

function diagnosticClip(label, result, className = "") {
  const clip = document.createElement("div");
  clip.className = `trigger-clip ${className}`.trim();
  const heading = document.createElement("div");
  heading.className = "trigger-clip-label";
  const phase = document.createElement("span");
  phase.textContent = label;
  const metric = document.createElement("strong");
  metric.textContent = diagnosticMetric(result);
  heading.append(phase, metric);
  clip.append(heading);
  if (result?.audio_url) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = result.audio_url;
    clip.append(audio);
  } else {
    clip.classList.add("waiting");
  }
  return clip;
}

function renderDiagnostics(events, capacity) {
  elements.triggerReviewList.replaceChildren();
  elements.triggerReviewSummary.textContent = events.length
    ? `${events.length} of ${capacity} recent trigger${events.length === 1 ? "" : "s"}. Review only mistakes; the oldest drops out automatically.`
    : `No triggers waiting for review. The queue keeps the latest ${capacity}.`;
  if (!events.length) {
    const empty = document.createElement("p");
    empty.className = "trigger-empty";
    empty.textContent = "Accepted Start phrases will appear here with the command that followed.";
    elements.triggerReviewList.append(empty);
    return;
  }

  events.forEach((event) => {
    const card = document.createElement("article");
    card.className = "trigger-event";
    const content = document.createElement("div");
    const head = document.createElement("div");
    head.className = "trigger-event-head";
    const title = document.createElement("strong");
    const predicted = event.command?.utterance || event.command?.best_utterance;
    title.textContent = predicted ? `Start → ${predicted}` : "Start → waiting for command";
    const timestamp = document.createElement("time");
    timestamp.dateTime = event.created_at || "";
    const created = new Date(event.created_at);
    timestamp.textContent = Number.isNaN(created.getTime())
      ? event.id
      : created.toLocaleString();
    head.append(title, timestamp);

    const pair = document.createElement("div");
    pair.className = "trigger-pair";
    const arrow = document.createElement("span");
    arrow.className = "trigger-arrow";
    arrow.textContent = "→";
    pair.append(
      diagnosticClip("Start accepted", event.start),
      arrow,
      diagnosticClip("Following command", event.command, "command"),
    );
    content.append(head, pair);

    const actions = document.createElement("div");
    actions.className = "trigger-actions";
    const teach = document.createElement("button");
    teach.type = "button";
    teach.className = "teach-rejection-button";
    teach.textContent = "Teach as false trigger";
    teach.addEventListener("click", () => markDiagnosticFalse(event.id, card));
    const dismiss = document.createElement("button");
    dismiss.type = "button";
    dismiss.className = "secondary-button dismiss-trigger-button";
    dismiss.textContent = "Dismiss";
    dismiss.addEventListener("click", () => dismissDiagnostic(event.id, card));
    actions.append(teach, dismiss);
    card.append(content, actions);
    elements.triggerReviewList.append(card);
  });
}

async function loadDiagnostics({ force = false } = {}) {
  if (state.diagnosticsLoading) return;
  state.diagnosticsLoading = true;
  elements.refreshTriggersButton.disabled = true;
  try {
    const response = await fetch("/api/diagnostics", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not load recent triggers");
    const events = payload.events || [];
    const signature = events
      .map((event) => `${event.id}:${event.command?.audio_url || "pending"}`)
      .join("|");
    if (force || signature !== state.diagnosticSignature) {
      state.diagnosticSignature = signature;
      renderDiagnostics(events, Number(payload.capacity) || 20);
    }
  } finally {
    state.diagnosticsLoading = false;
    elements.refreshTriggersButton.disabled = false;
  }
}

async function markDiagnosticFalse(eventId, card) {
  if (!window.confirm("Teach both clips as examples that must be rejected?")) return;
  card.querySelectorAll("button").forEach((button) => { button.disabled = true; });
  try {
    const response = await fetch(
      `/api/diagnostics/${encodeURIComponent(eventId)}/false-trigger`,
      { method: "POST" },
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not teach this rejection");
    state.diagnosticSignature = "";
    await loadDiagnostics({ force: true });
    setMessage("False trigger added to the rejection library; the listener was reloaded.", "success");
  } catch (error) {
    card.querySelectorAll("button").forEach((button) => { button.disabled = false; });
    setMessage(error.message, "error");
  }
}

async function dismissDiagnostic(eventId, card) {
  card.querySelectorAll("button").forEach((button) => { button.disabled = true; });
  try {
    const response = await fetch(`/api/diagnostics/${encodeURIComponent(eventId)}`, {
      method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not dismiss this trigger");
    state.diagnosticSignature = "";
    await loadDiagnostics({ force: true });
    setMessage("Trigger dismissed.");
  } catch (error) {
    card.querySelectorAll("button").forEach((button) => { button.disabled = false; });
    setMessage(error.message, "error");
  }
}

function selectCommand(index) {
  if (state.recording || state.continuousRunning || state.batchRunning) return;
  clearReplacement();
  discardReview();
  state.testMode = false;
  state.activeIndex = index;
  renderCommands();
  renderActiveCommand();
  const command = activeCommand();
  setMessage(command?.is_negative
    ? "Say a different short phrase each time—never one of the commands."
    : "Ready for a natural speaking voice.");
  loadSavedTakes().catch((error) => setMessage(error.message, "error"));
}

function selectTestMode() {
  if (state.recording || state.continuousRunning || state.batchRunning) return;
  clearReplacement();
  discardReview();
  state.testMode = true;
  renderCommands();
  renderActiveCommand();
  setMessage("Say one command naturally. No external action will run.");
  elements.recordButton.focus();
}

function canvasContext() {
  const canvas = elements.waveform;
  const scale = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * scale));
  const height = Math.max(1, Math.floor(canvas.clientHeight * scale));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { context: canvas.getContext("2d"), width, height, scale };
}

function drawIdleWave() {
  const { context, width, height, scale } = canvasContext();
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#315cda";
  context.lineWidth = scale;
  context.beginPath();
  context.moveTo(0, height / 2);
  context.lineTo(width, height / 2);
  context.stroke();
  elements.levelFill.style.width = "0%";
  elements.levelValue.textContent = "—";
}

async function drawCapturedWave(blob) {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContextClass();
    const decoded = await audioContext.decodeAudioData(await blob.arrayBuffer());
    const samples = decoded.getChannelData(0);
    const { context, width, height, scale } = canvasContext();
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "#315cda";
    context.lineWidth = 2 * scale;
    context.beginPath();
    let sum = 0;
    const bucket = Math.max(1, Math.floor(samples.length / width));
    for (let x = 0; x < width; x += 1) {
      let peak = 0;
      const start = x * bucket;
      const end = Math.min(samples.length, start + bucket);
      for (let index = start; index < end; index += 1) {
        peak = Math.max(peak, Math.abs(samples[index]));
        sum += samples[index] * samples[index];
      }
      const top = height / 2 - peak * height * 0.42;
      const bottom = height / 2 + peak * height * 0.42;
      context.moveTo(x, top);
      context.lineTo(x, bottom);
    }
    context.stroke();
    const rms = Math.sqrt(sum / Math.max(1, samples.length));
    const db = rms > 0 ? 20 * Math.log10(rms) : -60;
    const level = Math.max(0, Math.min(100, ((db + 55) / 45) * 100));
    elements.levelFill.style.width = `${level}%`;
    elements.levelFill.style.background = level > 88 ? "#dd554d" : level > 58 ? "#e8aa34" : "#42c5b3";
    elements.levelValue.textContent = `${Math.round(db)} dB`;
    await audioContext.close();
  } catch {
    drawIdleWave();
  }
}

async function startRecording() {
  if (state.recording || state.reviewBlob || state.continuousRunning || state.batchRunning || !elements.deviceSelect.value) return;
  elements.recordButton.disabled = true;
  elements.compareMicsButton.disabled = true;
  state.recording = true;
  renderBatchAvailability();
  elements.listeningLane.dataset.mode = "recording";
  elements.recordLabel.textContent = "Recording…";
  setSession("Recording", "live");
  setMessage(state.replaceFilename
    ? `Speak now to replace ${state.replaceTakeLabel}. The saved take remains until you keep this one.`
    : "Speak now. The local recorder will capture three seconds.");

  try {
    const response = await fetch("/api/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device: Number(elements.deviceSelect.value) }),
    });
    if (!response.ok) {
      const payload = await response.json();
      throw new Error(payload.error || "Microphone capture failed");
    }
    const blob = await response.blob();
    if (state.testMode) await testRecording(blob);
    else await showReview(blob);
  } catch (error) {
    state.recording = false;
    renderBatchAvailability();
    elements.listeningLane.dataset.mode = "idle";
    elements.recordButton.disabled = false;
    elements.compareMicsButton.disabled = comparableDevices().length < 2;
    elements.recordLabel.textContent = "Record sample";
    setSession("Ready", "ready");
    setMessage(error.message, "error");
  }
}

function renderContinuousResult(payload) {
  state.continuousNoticeUntil = Date.now() + 3000;
  if (payload.kind === "start_phrase") {
    state.continuousAwaitingCommand = true;
    state.continuousCommandDeadline = Date.now() + state.commandTimeoutSeconds * 1000;
    elements.phrase.textContent = "Start phrase heard";
    elements.description.textContent = `Say one command within ${state.commandTimeoutSeconds} seconds.`;
    setSession("Command window", "live");
    setMessage("Ready for one command.");
    return;
  }
  if (payload.kind === "ignored") {
    if (payload.rejection_reason === "duration_limit") {
      elements.phrase.textContent = "Start phrase not recognized";
      elements.description.textContent = "The capture reached 2.5 seconds and was treated as continuous background audio.";
      setSession("Background audio rejected", "warning");
      setMessage("Long continuous audio was ignored—say the Start phrase again.", "error");
      return;
    }
    if (payload.rejection_reason === "start_too_short") {
      elements.phrase.textContent = "Start phrase too short";
      elements.description.textContent = `Only ${payload.audio_seconds.toFixed(2)} seconds of speech was captured; at least ${payload.min_audio_seconds.toFixed(2)} seconds is required.`;
      setSession("Short sound rejected", "warning");
      setMessage("A short background sound was ignored—say the full Start phrase.", "error");
      return;
    }
    if (payload.rejection_reason === "start_too_long") {
      elements.phrase.textContent = "Start phrase too long";
      elements.description.textContent = `The capture lasted ${payload.audio_seconds.toFixed(2)} seconds; the Start phrase limit is ${payload.max_audio_seconds.toFixed(2)} seconds.`;
      setSession("Long sound rejected", "warning");
      setMessage("Long speech was ignored—say only the Start phrase.", "error");
      return;
    }
    elements.phrase.textContent = "Start phrase not recognized";
    elements.description.textContent = `Speech was detected, but it did not match the Start phrase (distance ${payload.score.toFixed(2)}, margin ${(payload.margin * 100).toFixed(1)}%).`;
    setSession("Start rejected", "warning");
    setMessage("Attempt rejected—say the Start phrase again.", "error");
    return;
  }
  state.continuousAwaitingCommand = false;
  state.continuousCommandDeadline = 0;
  if (payload.rejection_reason === "command_too_short") {
    elements.phrase.textContent = "Command too short";
    elements.description.textContent = `Only ${payload.audio_seconds.toFixed(2)} seconds of speech was captured; at least ${payload.min_audio_seconds.toFixed(2)} seconds is required.`;
    setSession("Short command rejected", "warning");
    setMessage("A short background sound was ignored—no action was run.", "error");
    return;
  }
  if (payload.accepted) {
    elements.phrase.textContent = payload.utterance;
    elements.description.textContent = `Recognized continuously at distance ${payload.score.toFixed(2)}. No action was run.`;
    setSession("Command matched", "ready");
    setMessage(`Matched “${payload.utterance}”. Say the Start phrase for another command.`, "success");
  } else {
    elements.phrase.textContent = "Command not recognized";
    elements.description.textContent = `Closest was “${payload.best_utterance}” at distance ${payload.score.toFixed(2)} with ${(payload.margin * 100).toFixed(1)}% separation; safely rejected.`;
    setSession("Command rejected", "warning");
    setMessage("Command attempt rejected—say the Start phrase to try again.", "error");
  }
}

function renderCommandTimeout() {
  state.continuousAwaitingCommand = false;
  state.continuousCommandDeadline = 0;
  state.continuousNoticeUntil = Date.now() + 3000;
  elements.phrase.textContent = "Command window expired";
  elements.description.textContent = `No command was received within ${state.commandTimeoutSeconds} seconds.`;
  setSession("Timed out", "warning");
  setMessage("Say the Start phrase to open a new command window.", "error");
}

function applyContinuousState(payload) {
  const wasRunning = state.continuousRunning;
  state.continuousRunning = Boolean(payload.running);
  elements.continuousButton.setAttribute("aria-pressed", state.continuousRunning ? "true" : "false");
  elements.continuousLabel.textContent = state.continuousRunning ? "Stop listening" : "Start continuous";
  elements.testButton.disabled = state.continuousRunning;
  elements.deviceSelect.disabled = state.continuousRunning || state.devices.length === 0;
  elements.refreshButton.disabled = state.continuousRunning;
  elements.recordButton.disabled = state.continuousRunning || state.devices.length === 0;
  elements.compareMicsButton.disabled = state.continuousRunning || comparableDevices().length < 2;
  renderBatchAvailability();

  if (state.continuousRunning) {
    if (!wasRunning) {
      clearReplacement();
      discardReview();
      state.testMode = true;
      renderCommands();
      renderActiveCommand();
      setMessage("Listening continuously. Speak a command, then pause briefly.");
      state.continuousNoticeUntil = 0;
      state.continuousAwaitingCommand = false;
      state.continuousCommandDeadline = 0;
    }
    elements.savedPlayback.pause();
    elements.savedTakes.classList.add("is-locked");
    elements.savedPlayback.setAttribute("aria-disabled", "true");
    elements.savedSelect.disabled = true;
    elements.replaceTakeButton.disabled = true;
    elements.listeningLane.dataset.mode = "continuous";
    elements.captureWindow.textContent = "voice activated";
    elements.recordLabel.textContent = "Listening continuously";
    const level = Math.max(0, Math.min(100, ((payload.level_db + 55) / 45) * 100));
    elements.levelFill.style.width = `${level}%`;
    elements.levelValue.textContent = `${Math.round(payload.level_db)} dB`;
    const hasNewEvent = payload.event_id > state.continuousEventId && payload.last_result;
    if (hasNewEvent) {
      state.continuousEventId = payload.event_id;
      renderContinuousResult(payload.last_result);
    }
    if (state.continuousAwaitingCommand && payload.phase === "waiting_for_start") {
      renderCommandTimeout();
    }

    if (payload.phase === "hearing") {
      setSession("Voice detected", "live");
      setMessage(state.continuousAwaitingCommand
        ? "Listening for the command—finish speaking, then pause."
        : "Listening—finish the Start phrase, then pause.");
    } else if (payload.phase === "matching") {
      setSession("Matching");
      setMessage(state.continuousAwaitingCommand
        ? "Comparing the command with your local examples…"
        : "Checking the Start phrase…");
    } else if (payload.phase === "awaiting_command") {
      setSession("Command window", "live");
      const remaining = Math.max(0, (state.continuousCommandDeadline - Date.now()) / 1000);
      elements.phrase.textContent = "Command window open";
      elements.description.textContent = `Say one command · ${remaining.toFixed(1)} seconds remaining.`;
      setMessage("Ready for one command.");
    } else if (payload.phase === "waiting_for_start") {
      if (Date.now() >= state.continuousNoticeUntil) {
        elements.phrase.textContent = "Waiting for Start phrase";
        elements.description.textContent = "Speak naturally, then pause while it checks the phrase.";
        setSession("Waiting", "ready");
        setMessage("Listening for the Start phrase.");
      }
    } else {
      setSession("Listening", "live");
    }
    if (payload.error) setMessage(payload.error, "error");
    return;
  }

  if (wasRunning) {
    elements.savedTakes.classList.remove("is-locked");
    elements.savedPlayback.removeAttribute("aria-disabled");
    elements.listeningLane.dataset.mode = "idle";
    elements.captureWindow.textContent = "3.0 sec";
    renderActiveCommand();
    drawIdleWave();
    setSession("Ready", "ready");
    setMessage(payload.error || "Continuous listening stopped.", payload.error ? "error" : "normal");
    loadSavedTakes().catch((error) => setMessage(error.message, "error"));
  }
}

async function pollContinuousState() {
  try {
    const response = await fetch("/api/listener", { cache: "no-store" });
    if (!response.ok) return;
    applyContinuousState(await response.json());
  } catch {
    // The next poll will retry; normal during a local server restart.
  }
}

async function toggleContinuous() {
  elements.continuousButton.disabled = true;
  try {
    const endpoint = state.continuousRunning ? "/api/listener/stop" : "/api/listener/start";
    const options = state.continuousRunning
      ? { method: "POST" }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device: Number(elements.deviceSelect.value) }),
        };
    const response = await fetch(endpoint, options);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not change listening mode");
    applyContinuousState(payload);
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    elements.continuousButton.disabled = false;
  }
}

async function testRecording(blob) {
  state.recording = false;
  renderBatchAvailability();
  await drawCapturedWave(blob);
  setSession("Matching");
  setMessage("Comparing this take with your local examples…");
  const response = await fetch("/api/match", {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: blob,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Recognition failed");
  elements.listeningLane.dataset.mode = "idle";
  elements.recordButton.disabled = false;
  elements.compareMicsButton.disabled = comparableDevices().length < 2;
  elements.recordLabel.textContent = "Test again";
  if (payload.accepted) {
    elements.phrase.textContent = payload.utterance;
    elements.description.textContent = `Recognized with distance ${payload.score.toFixed(2)}. No action was run.`;
    setSession("Matched", "ready");
    setMessage(`Matched “${payload.utterance}”.`, "success");
  } else {
    elements.phrase.textContent = "No command matched";
    elements.description.textContent = `Closest was “${payload.best_utterance}” at ${payload.score.toFixed(2)}, outside the safe boundary.`;
    setSession("Rejected");
    setMessage("Safely rejected—nothing would run.");
  }
}

async function showReview(blob) {
  state.recording = false;
  state.reviewBlob = blob;
  renderBatchAvailability();
  state.reviewUrl = URL.createObjectURL(blob);
  elements.listeningLane.dataset.mode = "review";
  elements.recordButton.disabled = true;
  elements.compareMicsButton.disabled = true;
  elements.recordLabel.textContent = "Review take";
  elements.playback.src = state.reviewUrl;
  elements.reviewPanel.hidden = false;
  elements.keepButton.textContent = state.replaceFilename ? "Replace saved take" : "Keep recording";
  setSession("Review");
  setMessage("Listen to the take, then keep it or retry.");
  await drawCapturedWave(blob);
  elements.playback.play().catch(() => {});
  elements.retryButton.focus();
}

function discardReview() {
  if (state.reviewUrl) URL.revokeObjectURL(state.reviewUrl);
  state.reviewBlob = null;
  state.reviewUrl = null;
  renderBatchAvailability();
  elements.playback.removeAttribute("src");
  elements.reviewPanel.hidden = true;
  elements.keepButton.textContent = "Keep recording";
  elements.listeningLane.dataset.mode = "idle";
  elements.recordButton.disabled = state.devices.length === 0;
  elements.compareMicsButton.disabled = comparableDevices().length < 2 || state.continuousRunning;
  elements.recordLabel.textContent = state.devices.length ? "Record sample" : "No microphone found";
  setSession(state.devices.length ? "Ready" : "No microphone", state.devices.length ? "ready" : "idle");
  drawIdleWave();
}

function clearReplacement() {
  state.replaceFilename = null;
  state.replaceTakeLabel = null;
}

function retryRecording() {
  discardReview();
  setMessage(state.replaceFilename
    ? `New take discarded. ${state.replaceTakeLabel} is unchanged; ready to try again.`
    : "Take discarded. Ready when you are.");
  elements.recordButton.focus();
}

function replaceSelectedTake() {
  if (state.recording || state.reviewBlob || state.continuousRunning) return;
  const option = elements.savedSelect.selectedOptions[0];
  if (!option?.dataset.filename) return;
  state.replaceFilename = option.dataset.filename;
  state.replaceTakeLabel = option.textContent;
  startRecording();
}

async function keepRecording() {
  if (!state.reviewBlob) return;
  const command = activeCommand();
  elements.keepButton.disabled = true;
  elements.retryButton.disabled = true;
  setMessage("Saving this take locally…");
  try {
    const endpoint = state.replaceFilename
      ? `/api/recordings/${encodeURIComponent(command.name)}/${encodeURIComponent(state.replaceFilename)}`
      : `/api/recordings/${encodeURIComponent(command.name)}`;
    const replacedTakeLabel = state.replaceTakeLabel;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: state.reviewBlob,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The recording could not be saved");
    command.count = payload.count;
    const replaced = Boolean(state.replaceFilename);
    clearReplacement();
    discardReview();
    renderCommands();
    renderActiveCommand();
    await loadSavedTakes();
    setMessage(replaced
      ? `Replaced ${replacedTakeLabel} with the new recording.`
      : command.is_negative
      ? `Saved non-command example ${command.count}.`
      : command.is_start_phrase
        ? `Saved start-phrase example ${command.count}.`
        : `Saved sample ${command.count} for “${command.utterance}”.`);
    if (!replaced && command.count >= command.target_samples) {
      if (command.is_start_phrase) {
        state.startPhraseReady = state.startPhraseEnabled;
        elements.continuousHint.textContent = state.startPhraseEnabled
          ? `Start phrase → one command within ${state.commandTimeoutSeconds} seconds. Listening stops when this page closes.`
          : "Start phrase samples complete. Calibrate before enabling the gate.";
      }
      const nextIndex = state.commands.findIndex((item) => item.count < item.target_samples);
      if (nextIndex >= 0) selectCommand(nextIndex);
      else setMessage("Training set complete. Ready for calibration.");
    }
    if (payload.listener_error) {
      setMessage(`Recording saved, but the listener could not reload: ${payload.listener_error}`, "error");
    }
    elements.recordButton.focus();
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    elements.keepButton.disabled = false;
    elements.retryButton.disabled = false;
  }
}

function openNewCommand() {
  if (!state.canManageCommands || state.recording || state.continuousRunning || state.batchRunning) return;
  elements.newCommandForm.reset();
  elements.newCommandTarget.value = "10";
  elements.newCommandDialog.showModal();
  elements.newCommandUtterance.focus();
}

async function createCommand(event) {
  event.preventDefault();
  const submit = elements.newCommandForm.querySelector('[type="submit"]');
  submit.disabled = true;
  try {
    const response = await fetch("/api/commands", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        utterance: elements.newCommandUtterance.value,
        description: elements.newCommandDescription.value,
        target_samples: Number(elements.newCommandTarget.value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The command could not be added");
    elements.newCommandDialog.close();
    await loadState();
    const index = state.commands.findIndex((command) => command.name === payload.command);
    if (index >= 0) selectCommand(index);
    setMessage(payload.listener_error
      ? `Command added, but the listener could not reload: ${payload.listener_error}`
      : `Added “${payload.utterance}”. Record examples when ready.`,
    payload.listener_error ? "error" : "success");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    submit.disabled = false;
  }
}

async function deleteSelectedTake() {
  if (state.recording || state.reviewBlob || state.continuousRunning || state.batchRunning) return;
  const command = activeCommand();
  const option = elements.savedSelect.selectedOptions[0];
  const filename = option?.dataset.filename;
  if (!command || !filename) return;
  if (!window.confirm(`Remove ${option.textContent} for “${command.utterance}”? It will remain recoverable on this computer.`)) return;
  elements.deleteTakeButton.disabled = true;
  try {
    const response = await fetch(
      `/api/recordings/${encodeURIComponent(command.name)}/${encodeURIComponent(filename)}`,
      { method: "DELETE" },
    );
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The recording could not be removed");
    command.count = payload.count;
    renderCommands();
    renderActiveCommand();
    await loadSavedTakes();
    setMessage(payload.listener_error
      ? `Take archived, but the listener could not reload: ${payload.listener_error}`
      : "Take removed from matching and saved in the recovery archive.",
    payload.listener_error ? "error" : "success");
  } catch (error) {
    setMessage(error.message, "error");
    elements.deleteTakeButton.disabled = false;
  }
}

async function deleteActiveCommand() {
  const command = activeCommand();
  if (!command || command.is_negative || command.is_start_phrase) return;
  if (!window.confirm(`Remove “${command.utterance}” and all of its active takes? They will remain recoverable on this computer.`)) return;
  elements.deleteCommandButton.disabled = true;
  try {
    const response = await fetch(`/api/commands/${encodeURIComponent(command.name)}`, {
      method: "DELETE",
      headers: { "X-Confirm-Command": command.name },
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The command could not be removed");
    state.activeIndex = 0;
    await loadState();
    setMessage(payload.listener_error
      ? `Command archived, but the listener could not reload: ${payload.listener_error}`
      : `Removed “${command.utterance}” from matching.`,
    payload.listener_error ? "error" : "success");
  } catch (error) {
    setMessage(error.message, "error");
    elements.deleteCommandButton.disabled = false;
  }
}

elements.refreshButton.addEventListener("click", async () => {
  elements.refreshButton.disabled = true;
  setMessage("Refreshing input devices…");
  try {
    await loadState({ preserveSelection: true });
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    elements.refreshButton.disabled = false;
  }
});
elements.deviceSelect.addEventListener("change", () => {
  window.localStorage.setItem("voiceprint-input-device", elements.deviceSelect.value);
});

elements.recordButton.addEventListener("click", startRecording);
elements.batchButton.addEventListener("click", openBatchRecorder);
elements.cancelBatchButton.addEventListener("click", closeBatchRecorder);
elements.startBatchButton.addEventListener("click", startBatchRecorder);
elements.retryBatchButton.addEventListener("click", retryBatchRecorder);
elements.stopBatchButton.addEventListener("click", stopBatchRecorder);
elements.doneBatchButton.addEventListener("click", closeBatchRecorder);
elements.batchDialog.addEventListener("cancel", (event) => {
  if (!state.batchRunning) return;
  event.preventDefault();
  stopBatchRecorder();
});
elements.retryButton.addEventListener("click", retryRecording);
elements.keepButton.addEventListener("click", keepRecording);
elements.replaceTakeButton.addEventListener("click", replaceSelectedTake);
elements.deleteTakeButton.addEventListener("click", deleteSelectedTake);
elements.deleteCommandButton.addEventListener("click", deleteActiveCommand);
elements.addCommandButton.addEventListener("click", openNewCommand);
elements.cancelCommandButton.addEventListener("click", () => elements.newCommandDialog.close());
elements.newCommandForm.addEventListener("submit", createCommand);
elements.testButton.addEventListener("click", selectTestMode);
elements.compareMicsButton.addEventListener("click", openMicrophoneComparison);
elements.closeCompareButton.addEventListener("click", closeMicrophoneComparison);
elements.cancelCompareButton.addEventListener("click", closeMicrophoneComparison);
elements.startCompareButton.addEventListener("click", startMicrophoneComparison);
elements.recordCompareButton.addEventListener("click", recordComparisonTake);
elements.restartCompareButton.addEventListener("click", restartMicrophoneComparison);
elements.finishCompareButton.addEventListener("click", closeMicrophoneComparison);
elements.micCompareDialog.addEventListener("close", clearComparisonRecordings);
elements.savedSelect.addEventListener("change", () => {
  elements.savedPlayback.src = elements.savedSelect.value;
  elements.savedPlayback.play().catch(() => {});
});
elements.continuousButton.addEventListener("click", toggleContinuous);
elements.refreshTriggersButton.addEventListener("click", () => {
  loadDiagnostics({ force: true }).catch((error) => setMessage(error.message, "error"));
});

document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "BUTTON", "AUDIO"].includes(document.activeElement?.tagName)) return;
  if (event.code === "Space" && !state.reviewBlob) {
    event.preventDefault();
    startRecording();
  } else if (event.key.toLowerCase() === "r" && state.reviewBlob) {
    retryRecording();
  } else if (event.key === "Enter" && state.reviewBlob) {
    keepRecording();
  }
});

window.addEventListener("resize", () => {
  if (state.reviewBlob) drawCapturedWave(state.reviewBlob);
  else drawIdleWave();
});

loadState().catch((error) => {
  setMessage(error.message, "error");
  elements.description.textContent = "Check that the local studio server is running.";
});
loadDiagnostics().catch((error) => setMessage(error.message, "error"));

setInterval(pollContinuousState, 400);
setInterval(() => {
  loadDiagnostics().catch(() => {});
}, 5000);
window.addEventListener("pagehide", () => {
  if (state.continuousRunning) navigator.sendBeacon("/api/listener/stop");
  if (state.batchRunning) navigator.sendBeacon("/api/batch/finish");
});
