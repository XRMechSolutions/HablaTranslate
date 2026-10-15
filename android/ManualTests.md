# ManualTests

This file is for owner-run full end-to-end verification after automated work
lands. The orchestrator should already have produced correct work through
self-gating audits and correction tasks. This file is not an AwakenedSpanreed
scheduler input and must not gate orchestrator progress.

## Scope

- Project: Habla Android (`android/`), standalone on-device translator (D002).
- Build/run command or app entry point: from `android/`,
  `gradlew.bat :app:installPublicDebug` (or `:app:installPersonalDebug` with a filled
  `android/.env`, D022), then open "Habla" on the phone.
- Required accounts, fixtures, devices, or environment: Samsung Galaxy A56 (D011) with
  USB debugging; an OpenAI API key with credit, plus a revoked or made-up key for M5; Wi-Fi
  and the ability to toggle airplane mode; a second phone to place a call (M8); a quiet room
  and a classroom or conversation partner (M1, M2). Speech model downloaded in Settings
  (264 MB) unless the run says otherwise.
- Record for every run: device, One UI version, build variant and `versionName`, date, and the
  Diagnostics metrics line from Settings at the end.
- Finished-app workflows covered: live listening (classroom and conversation), screen-off
  listening, failures (network, key, model download, calls, mic toggle, notifications denied,
  thermal), Samsung battery behaviour, app lifecycle, personal-flavor key seeding, backup.

Planning sources: `.dev/planning/modules/app/design.md` section 11 (Manual tests),
`.dev/planning/modules/app/edge-cases.md`, `.dev/planning/vision.md` success criteria.

## End-to-End Runs

### M1 Classroom session, screen on, 45 minutes

- [ ] Partials and finals keep up for 45 minutes and cost stays inside the target.
  - Setup: key saved, model Ready, mode Classroom, direction ES->EN, phone on a desk, screen on.
  - Steps: tap Listen, leave it running through a 45-minute Spanish class, tap Stop. Open
    Settings > Diagnostics.
  - Expected: cards keep appearing in order with translations; a "Live" partial line updates
    while the teacher speaks; no FALLING_BEHIND or AUDIO_DROPPED banner; Diagnostics shows
    partial cadence p50 <= 1.5 s, speech-end-to-translation p50 near 3 s, and session cost
    about $0.02/min or less; the session appears in History with all exchanges.
  - Failure signal: partial line freezes, cards arrive minutes late, "Phone falling behind"
    banner, cost per minute well above $0.02.

### M2 Two-language conversation, direction Auto

- [ ] Direction flips per speaker turn.
  - Setup: direction Auto, mode Conversation, built-in mic (headsets change parity).
  - Steps: alternate Spanish and English turns for 10 minutes with a partner.
  - Expected: Spanish turns show ES->EN cards, English turns EN->ES; the direction selector
    reads "Auto (ES)" or "Auto (EN)" following the voted language.
  - Failure signal: every card stays in one direction, or English is "translated" into English.

### M3 Screen off for 60 minutes, unplugged

- [ ] Listening continues with the screen off and nothing is lost.
  - Setup: unplugged, battery % noted, key and model ready.
  - Steps: tap Listen, turn the screen off, let speech run for 60 minutes, unlock, Stop.
  - Expected: exchanges cover the whole hour with no gaps; Diagnostics `frames_dropped` is 0;
    battery drop recorded from the metrics line; notification read "Listening - N exchanges".
  - Failure signal: exchanges stop after some minutes with the screen off, or `frames_dropped` > 0.

### M4 Network loss mid-session

- [ ] Transcription continues offline and translation recovers afterwards.
  - Steps: during a session switch airplane mode on for 2 minutes, then off.
  - Expected: OFFLINE banner "Offline - transcribing only; translations fail until back online";
    cards during the outage read "Not translated - network: <message>" but keep their
    transcripts; after reconnecting new cards translate and the banner clears.
  - Failure signal: the session stops, transcripts disappear, or translations never resume.

### M5 Bad key, then good key

- [ ] Key errors are explained and the Test key texts match.
  - Steps: Settings: save a made-up key, tap Test key; start a session and speak; then save the
    real key and Test key again; speak again.
  - Expected: Test key shows "Key rejected by OpenAI (401)"; Live shows KEY_INVALID "OpenAI
    rejected the key (401) - check or replace it" with a Settings action; cards read "Not
    translated - key rejected"; after the good key "Key works (<model> replied)" and new cards
    translate; the banner clears. The key text is never shown after saving.
  - Failure signal: generic error text, banner never clears, key visible in the field.

### M6 Model download interrupted

- [ ] Download resumes after airplane mode and after force stop and verifies the hash.
  - Setup: Settings > Speech model > Delete.
  - Steps: Download; at ~30% turn airplane mode on, then off and tap Retry; at ~60% force-stop
    the app in Android settings, reopen and tap Download.
  - Expected: MODEL_FAILED banner with the network reason, then resume from the partial size
    (not from 0); after force stop the progress resumes; ends in Ready after "Verifying".
  - Failure signal: restarts from 0 MB, stuck at Verifying, or Ready with a corrupt model
    (session fails with "Speech model failed to load").

### M7 Battery and thermal, 30 minutes

- [ ] Thermal pause appears and clears; finals never stop.
  - Steps: listen 30 minutes in a warm place (or charging while gaming beforehand).
  - Expected: if the phone reaches severe thermal status the banner "Phone is hot - live
    partials paused" appears, partials stop, finals continue; it clears when the phone cools;
    the metrics line lists thermal changes and battery start/end %.
  - Failure signal: finals stop, or partials never resume after cooling.

### M8 Incoming call during a session

- [ ] Call silences capture and the session resumes by itself.
  - Steps: while listening, receive and answer a call from another phone, talk 1 minute, hang up.
  - Expected: MIC_SILENCED banner "Microphone in use by another app or muted - paused" and the
    notification "Mic silenced by another app"; no garbage cards during the call; capture
    resumes after hang-up without tapping anything.
  - Failure signal: session ends, or it never resumes after the call.

### M9 Microphone privacy toggle

- [ ] Toggling the mic off pauses and toggling it on resumes.
  - Steps: during a session turn the quick-settings microphone access off for 30 s, then on.
  - Expected: MIC_SILENCED within about 2 s; clears on the first sound after re-enabling.
  - Failure signal: no banner, or false segments/cards appear while muted.

### M10 Notifications denied

- [ ] The session works without notification permission and Stop is in the app.
  - Setup: fresh install (or revoke notifications in Android settings).
  - Steps: deny notifications on the setup card, tap Listen, speak, tap Stop on Live.
  - Expected: setup row "Notifications off - stop listening from the app"; the session runs
    and stops normally; the status-bar mic indicator turns off within about a second of Stop.
  - Failure signal: Listen fails to start, or the mic indicator stays on.

### M11 Samsung sleeping apps and battery settings

- [ ] Background listening survives Samsung battery management.
  - Steps: Settings > Battery row opens the battery-optimisation screen; leave Habla at the
    default ("Optimized") and run M3 for 30 minutes; repeat with Habla set to "Unrestricted".
    Check Device care > Battery > Background usage limits for Habla in "Sleeping apps".
  - Expected: listening continues in both runs; record whether "Optimized" caused any gap.
  - Failure signal: the session silently stops or the service is killed with the screen off.

### M12 App lifecycle: swipe away and force stop

- [ ] Swiping from recents keeps listening; force stop ends it without a restart.
  - Steps: listen, swipe Habla away from recents, speak 1 minute, reopen; then force-stop it
    from Android settings and wait 2 minutes; reopen.
  - Expected: after the swipe the notification stays and new exchanges appear on reopen; after
    force stop the notification and mic indicator disappear and the service does not come back
    by itself; reopening shows Idle with the earlier session in History.
  - Failure signal: listening stops on swipe, or the service restarts on its own.

### M13 Personal flavor seeds once

- [ ] The personal build arrives with the key saved, and Clear key is not undone.
  - Setup: `android/.env` with the owner's key; `gradlew.bat :app:installPersonalDebug`.
  - Steps: open the app (fresh install), check Settings; tap Clear key and confirm; force-stop
    and reopen.
  - Expected: first open shows "Key saved" without typing; after Clear and restart it shows
    "No key" (not re-seeded). Never share this APK.
  - Failure signal: no key on first open, or the key reappears after Clear.

### M14 Backup and restore leave no key

- [ ] A restored backup brings back history but not the key or the model.
  - Steps: with a key saved and some history, run `adb shell bmgr backupnow com.xrmech.habla.debug`,
    uninstall, reinstall, `adb shell bmgr restore` (or restore through a new-device transfer).
  - Expected: History is present; Settings shows "No key" and the setup card asks for a key;
    the speech model shows Missing (MODEL_MISSING banner).
  - Failure signal: the key is present after restore, or the app crashes reading the secrets file.

### Negative or Failure Run: no key, no model

- [ ] The app stays usable and explains what is missing.
  - Setup: fresh public install; skip the key; do not download the model.
  - Steps: tap Listen; then download the model and Listen again with no key.
  - Expected: first Listen does nothing but scroll to "Speech model not downloaded (264 MB) -
    needed to transcribe" with Download; after download, cards show transcripts with "Not
    translated - no OpenAI key" and the NO_KEY banner "Translation off - add your OpenAI key".
  - Failure signal: a crash, a silent no-op, or a generic error.

## Not Covered

- Golden parity of segmentation, preprocessing and ASR (automated: `:audio`/`:asr` JVM tests and
  the emulator `PipelineSmokeTest`).
- Coordinator scenarios (automated: `PipelineCoordinatorTest`, 12 scenarios).
- Public APK key absence (automated: CI APK scan with a positive control).
- WER comparison of phone vs server (`:asr` D6 / `server` engine check).
