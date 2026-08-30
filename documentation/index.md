---
hide:
  - navigation
  - toc
---

<div class="lv-hero">
  <div>
    <p class="lv-kicker">Offline · trainable · cross-platform</p>
    <h1>A voice interface that stays home.</h1>
    <p class="lv-hero-copy">
      Teach your own short commands, recognize them locally, and connect the result
      to any automation. No transcription, cloud account, or fixed language.
    </p>
    <div class="lv-actions">
      <a class="lv-button lv-button-primary" href="getting-started/">Train your first command</a>
      <a class="lv-button" href="https://github.com/hkchengrex/home-assistant-voice-io">View the repository</a>
    </div>
  </div>
  <div class="lv-signal" aria-label="Example recognition: lights on, accepted in 87 milliseconds">
    <div class="lv-signal-head">
      <span class="lv-signal-label">Local signal / 16 kHz</span>
      <span class="lv-live">listening</span>
    </div>
    <div class="lv-wave" aria-hidden="true">
      <span style="--i:1;--h:18"></span><span style="--i:2;--h:30"></span>
      <span style="--i:3;--h:54"></span><span style="--i:4;--h:82"></span>
      <span style="--i:5;--h:44"></span><span style="--i:6;--h:68"></span>
      <span style="--i:7;--h:94"></span><span style="--i:8;--h:62"></span>
      <span style="--i:9;--h:38"></span><span style="--i:10;--h:74"></span>
      <span style="--i:11;--h:48"></span><span style="--i:12;--h:26"></span>
      <span style="--i:13;--h:58"></span><span style="--i:14;--h:34"></span>
      <span style="--i:15;--h:20"></span>
    </div>
    <div class="lv-signal-foot">
      <span><strong>Lights on</strong><br><small>command accepted</small></span>
      <span class="lv-signal-score">87 ms</span>
    </div>
  </div>
</div>

<div class="lv-flow">
  <div class="lv-step">
    <span class="lv-step-number">01 / Train</span>
    <h3>Say it your way</h3>
    <p>Record examples in the browser-based Studio. Any language or short phrase works.</p>
  </div>
  <div class="lv-step">
    <span class="lv-step-number">02 / Listen</span>
    <h3>Recognize locally</h3>
    <p>Use a wake phrase or listen directly. Audio stays on the machine doing the matching.</p>
  </div>
  <div class="lv-step">
    <span class="lv-step-number">03 / Act</span>
    <h3>Connect anything</h3>
    <p>Map an accepted command to Home Assistant, another automation system, or your own code.</p>
  </div>
</div>

## Small, focused, useful

<p class="lv-section-lead">
Local Voice Pipeline is for repeatable commands—not open-ended dictation. That narrow
job makes it private, understandable, and practical on modest hardware.
</p>

<div class="lv-capabilities">
  <div class="lv-capability">
    <h3>Training Studio</h3>
    <p>Add phrases, compare microphones, record hands-free sets, replay takes, and review recent triggers in a local web interface.</p>
    <p><a href="training-studio/">Open the Studio guide →</a></p>
  </div>
  <div class="lv-capability">
    <h3>Wake phrase or direct mode</h3>
    <p>Gate commands behind a trained start phrase, or classify every utterance while testing.</p>
    <p><a href="running/">Choose a listening mode →</a></p>
  </div>
  <div class="lv-capability">
    <h3>Automation-neutral</h3>
    <p>The recognizer returns intent names through a small handler contract. Your integration decides what happens next.</p>
    <p><a href="integrations/">Connect an automation →</a></p>
  </div>
  <div class="lv-capability">
    <h3>Windows, macOS, and Linux</h3>
    <p>The same Python package and configuration travel across platforms, with service examples for each.</p>
    <p><a href="deployment/">Deploy it →</a></p>
  </div>
</div>

<div class="lv-benchmark">
  <div>
    <span class="lv-kicker">Reference benchmark</span>
    <h2>Responsive on a first-generation Surface Go.</h2>
    <p>One 0.72-second command matched against 123 templates on an Intel Pentium Gold 4415Y.</p>
    <a href="benchmark/">See the method and full result →</a>
  </div>
  <div class="lv-stat">
    <span class="lv-stat-value">86.9</span>
    <span class="lv-stat-label">milliseconds median</span>
  </div>
</div>

!!! info "What this is not"
    This project does not transcribe arbitrary speech or answer open-ended questions.
    It recognizes the phrases you train and hands an intent to your application.
