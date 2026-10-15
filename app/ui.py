"""Modern Google Material 3 UI for the QUAN-ASR Server playground & developer demo.

Served at ``/``. Features:
- Google Material Design 3 design system (Light / Dark themes, Google colors, typography)
- Dual-mode File Transcription (Standard POST /v1/audio/transcriptions & real-time SSE streaming)
- 1-Click Vietnamese Audio Demo Samples
- Interactive Subtitle / Timestamp Timeline Viewer
- Real-time Live Microphone Streaming via WebSocket with Canvas Waveform Visualizer
- Observability Dashboard (Model status, Device, Realtime Factor, Metrics polling)
- API Integration Hub (Direct links to Scalar /scalar, Swagger /docs, Python SDK & cURL code generator)
"""

from __future__ import annotations

INDEX_HTML = r"""<!doctype html>
<html lang="vi" data-theme="dark">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>QUAN-ASR Studio · Nhận diện giọng nói tiếng Việt chuẩn Google</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%231a73e8'><path d='M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5-3c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z'/></svg>" />

<!-- Google Fonts & Material Symbols -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Google+Sans+Text:wght@400;500;700&family=JetBrains+Mono:wght@400;500&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />

<style>
  :root {
    --google-blue: #1a73e8;
    --google-blue-hover: #1557b0;
    --google-red: #ea4335;
    --google-yellow: #fbbc04;
    --google-green: #34a853;

    /* Light Theme (Google Standard) */
    --md-bg: #f8f9fa;
    --md-surface: #ffffff;
    --md-surface-variant: #f1f3f4;
    --md-surface-container: #ffffff;
    --md-surface-container-high: #e8eaed;
    --md-border: #dadce0;
    --md-border-subtle: #e8eaed;
    --md-text: #202124;
    --md-text-secondary: #5f6368;
    --md-text-muted: #80868b;
    --md-primary: #1a73e8;
    --md-primary-container: #e8f0fe;
    --md-on-primary-container: #174ea6;
    --md-shadow-1: 0 1px 3px 0 rgba(60,64,67,0.12), 0 1px 2px 0 rgba(60,64,67,0.24);
    --md-shadow-2: 0 2px 6px 2px rgba(60,64,67,0.15), 0 1px 2px 0 rgba(60,64,67,0.30);
    --md-code-bg: #f1f3f4;
  }

  [data-theme="dark"] {
    /* Dark Theme (Google AI Studio / Gemini Dark) */
    --md-bg: #131314;
    --md-surface: #1e1f20;
    --md-surface-variant: #282a2c;
    --md-surface-container: #1e1f20;
    --md-surface-container-high: #333537;
    --md-border: #3c4043;
    --md-border-subtle: #2d3135;
    --md-text: #e3e3e3;
    --md-text-secondary: #9aa0a6;
    --md-text-muted: #70757a;
    --md-primary: #8ab4f8;
    --md-primary-container: #174ea6;
    --md-on-primary-container: #d2e3fc;
    --md-shadow-1: 0 1px 3px 0 rgba(0,0,0,0.5), 0 1px 2px 0 rgba(0,0,0,0.4);
    --md-shadow-2: 0 4px 12px 0 rgba(0,0,0,0.6);
    --md-code-bg: #18191b;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: "Google Sans Text", "Roboto", system-ui, -apple-system, sans-serif;
    background-color: var(--md-bg);
    color: var(--md-text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    transition: background-color 0.25s ease, color 0.25s ease;
    line-height: 1.5;
  }

  /* Material Symbols utility */
  .icon {
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    font-size: 20px;
    line-height: 1;
    display: inline-block;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: normal;
    white-space: nowrap;
    direction: ltr;
    vertical-align: middle;
  }

  /* Header */
  header.google-app-bar {
    position: sticky;
    top: 0;
    z-index: 100;
    background: var(--md-surface);
    border-bottom: 1px solid var(--md-border);
    padding: 0 24px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: background 0.25s ease, border-color 0.25s ease;
  }

  .brand-group {
    display: flex;
    align-items: center;
    gap: 12px;
    text-decoration: none;
    color: inherit;
  }

  .logo-box {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #1a73e8, #4285f4);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    box-shadow: 0 2px 6px rgba(26,115,232,0.35);
  }

  .brand-title {
    font-family: 'Google Sans', sans-serif;
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.2px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .brand-tag {
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 9999px;
    background: var(--md-surface-variant);
    color: var(--md-text-secondary);
    border: 1px solid var(--md-border);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  /* Pills & Badges */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
    background: var(--md-surface-variant);
    color: var(--md-text-secondary);
    border: 1px solid var(--md-border);
    transition: all 0.2s ease;
  }

  .badge.status-ready {
    background: rgba(52, 168, 83, 0.12);
    border-color: rgba(52, 168, 83, 0.4);
    color: var(--google-green);
  }
  .badge.status-loading {
    background: rgba(251, 188, 4, 0.12);
    border-color: rgba(251, 188, 4, 0.4);
    color: var(--google-yellow);
  }
  .badge.status-error {
    background: rgba(234, 67, 53, 0.12);
    border-color: rgba(234, 67, 53, 0.4);
    color: var(--google-red);
  }

  .pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: currentColor;
    display: inline-block;
    animation: pulse 2s infinite ease-in-out;
  }

  @keyframes pulse {
    0%, 100% { transform: scale(0.9); opacity: 0.7; }
    50% { transform: scale(1.3); opacity: 1; }
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .spin {
    animation: spin 1s linear infinite;
    display: inline-block;
  }

  /* Model Status Banners */
  .model-banner-container {
    max-width: 1200px;
    width: 100%;
    margin: 16px auto 0 auto;
    padding: 0 24px;
    box-sizing: border-box;
  }
  .model-banner-card {
    border-radius: 14px;
    padding: 16px 20px;
    display: flex;
    align-items: flex-start;
    gap: 16px;
    box-shadow: var(--md-shadow-1);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
  }
  .banner-card-loading {
    background: rgba(26, 115, 232, 0.08);
    border: 1px solid rgba(26, 115, 232, 0.35);
  }
  [data-theme="dark"] .banner-card-loading {
    background: rgba(138, 180, 248, 0.08);
    border: 1px solid rgba(138, 180, 248, 0.3);
  }
  .banner-card-error {
    background: rgba(234, 67, 53, 0.08);
    border: 1px solid rgba(234, 67, 53, 0.35);
  }
  .banner-icon-col {
    display: flex;
    align-items: center;
    justify-content: center;
    padding-top: 2px;
  }
  .banner-spinner {
    width: 24px;
    height: 24px;
    border: 3px solid rgba(26, 115, 232, 0.2);
    border-top-color: var(--google-blue);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  [data-theme="dark"] .banner-spinner {
    border: 3px solid rgba(138, 180, 248, 0.2);
    border-top-color: var(--md-primary);
  }
  .banner-body {
    flex: 1;
    min-width: 0;
  }
  .banner-headline {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
    flex-wrap: wrap;
  }
  .banner-title {
    font-family: 'Google Sans', sans-serif;
    font-weight: 700;
    font-size: 15px;
    color: var(--md-text);
  }
  .banner-description {
    font-size: 13px;
    color: var(--md-text-secondary);
    line-height: 1.45;
  }
  .banner-progress-track {
    margin-top: 10px;
    height: 4px;
    width: 100%;
    background: rgba(26, 115, 232, 0.15);
    border-radius: 2px;
    overflow: hidden;
    position: relative;
  }
  .banner-progress-bar {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: 35%;
    background: var(--google-blue);
    border-radius: 2px;
    animation: banner-progress 1.6s infinite ease-in-out;
  }
  @keyframes banner-progress {
    0% { left: -35%; }
    50% { left: 40%; width: 50%; }
    100% { left: 100%; width: 25%; }
  }
  .sample-chip.disabled-loading {
    opacity: 0.6;
    cursor: not-allowed;
  }
  #micBtn.disabled-loading {
    opacity: 0.45;
    cursor: not-allowed;
    filter: grayscale(0.6);
  }

  .nav-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    text-decoration: none;
    color: var(--md-text-secondary);
    background: transparent;
    border: 1px solid var(--md-border);
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .nav-btn:hover {
    color: var(--md-primary);
    background: var(--md-surface-variant);
    border-color: var(--md-primary);
  }

  .icon-btn {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    border: 1px solid var(--md-border);
    background: var(--md-surface);
    color: var(--md-text);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .icon-btn:hover {
    background: var(--md-surface-variant);
    border-color: var(--md-primary);
    color: var(--md-primary);
  }

  /* Main Container */
  main {
    max-width: 1100px;
    width: 100%;
    margin: 0 auto;
    padding: 24px 20px 80px;
    flex: 1;
  }

  /* Google Tab Navigation Bar */
  .google-tabs {
    display: flex;
    border-bottom: 1px solid var(--md-border);
    gap: 8px;
    margin-bottom: 24px;
    overflow-x: auto;
  }

  .tab-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 18px;
    font-family: 'Google Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    color: var(--md-text-secondary);
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
    border-radius: 8px 8px 0 0;
  }

  .tab-item:hover {
    color: var(--md-text);
    background: var(--md-surface-variant);
  }

  .tab-item.active {
    color: var(--md-primary);
    border-bottom-color: var(--md-primary);
  }

  .tab-content {
    display: none;
    animation: fadeIn 0.25s ease;
  }
  .tab-content.active {
    display: block;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Cards & Surface Containers */
  .card {
    background: var(--md-surface);
    border: 1px solid var(--md-border);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: var(--md-shadow-1);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .card-title {
    font-family: 'Google Sans', sans-serif;
    font-size: 16px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--md-text);
  }

  .card-subtitle {
    font-size: 13px;
    color: var(--md-text-secondary);
    margin-top: 2px;
  }

  /* Sample Picker Banner */
  .samples-banner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: var(--md-surface-variant);
    border: 1px solid var(--md-border);
    border-radius: 12px;
    margin-bottom: 18px;
  }
  .samples-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--md-text-secondary);
  }
  .sample-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 500;
    background: var(--md-surface);
    border: 1px solid var(--md-border);
    color: var(--md-primary);
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .sample-chip:hover {
    background: var(--md-primary-container);
    color: var(--md-on-primary-container);
    border-color: var(--md-primary);
    transform: translateY(-1px);
  }

  /* Dropzone */
  .dropzone {
    border: 2px dashed var(--md-border);
    border-radius: 14px;
    padding: 30px 20px;
    text-align: center;
    background: var(--md-bg);
    cursor: pointer;
    transition: all 0.2s ease;
    margin-bottom: 18px;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--md-primary);
    background: var(--md-surface-variant);
  }
  .dropzone-icon {
    font-size: 44px;
    color: var(--md-primary);
    margin-bottom: 8px;
  }
  .dropzone-text {
    font-size: 14px;
    font-weight: 500;
    color: var(--md-text);
  }
  .dropzone-sub {
    font-size: 12px;
    color: var(--md-text-muted);
    margin-top: 4px;
  }

  /* Audio player */
  .audio-preview-box {
    display: none;
    align-items: center;
    gap: 16px;
    padding: 12px 16px;
    background: var(--md-surface-variant);
    border: 1px solid var(--md-border);
    border-radius: 12px;
    margin-bottom: 18px;
  }
  .audio-preview-box.has-file {
    display: flex;
    flex-wrap: wrap;
  }
  .audio-preview-box audio {
    flex: 1 1 280px;
    height: 38px;
    outline: none;
  }
  .file-meta-pill {
    font-size: 12px;
    color: var(--md-text-secondary);
    display: flex;
    align-items: center;
    gap: 6px;
  }

  /* Form Grid */
  .form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 18px;
  }

  .field-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  label.field-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--md-text-secondary);
  }

  select.google-input, input.google-input {
    width: 100%;
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid var(--md-border);
    background: var(--md-bg);
    color: var(--md-text);
    font-family: inherit;
    font-size: 13px;
    outline: none;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  select.google-input:focus, input.google-input:focus {
    border-color: var(--md-primary);
    box-shadow: 0 0 0 2px var(--md-primary-container);
  }

  /* Action Buttons */
  .actions-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    margin-top: 14px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 8px;
    font-family: 'Google Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    white-space: nowrap;
  }

  .btn-primary {
    background: var(--google-blue);
    color: #ffffff;
    box-shadow: 0 1px 3px rgba(26,115,232,0.3);
  }
  .btn-primary:hover:not(:disabled) {
    background: var(--google-blue-hover);
    box-shadow: 0 2px 6px rgba(26,115,232,0.45);
  }

  .btn-tonal {
    background: var(--md-primary-container);
    color: var(--md-on-primary-container);
    border-color: var(--md-border);
  }
  .btn-tonal:hover:not(:disabled) {
    background: var(--md-surface-variant);
    border-color: var(--md-primary);
  }

  .btn-outlined {
    background: transparent;
    border: 1px solid var(--md-border);
    color: var(--md-text);
  }
  .btn-outlined:hover:not(:disabled) {
    background: var(--md-surface-variant);
    border-color: var(--md-primary);
    color: var(--md-primary);
  }

  .btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  /* Accordion for advanced settings */
  .accordion-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 500;
    color: var(--md-primary);
    background: none;
    border: none;
    cursor: pointer;
    padding: 8px 0;
    margin: 8px 0;
  }
  .accordion-content {
    display: none;
    padding: 14px;
    background: var(--md-surface-variant);
    border: 1px solid var(--md-border);
    border-radius: 12px;
    margin-bottom: 16px;
  }
  .accordion-content.expanded {
    display: block;
    animation: fadeIn 0.2s ease;
  }

  /* Switch Toggle */
  .switch-label {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    font-size: 13px;
    color: var(--md-text);
  }
  .switch-box {
    width: 38px;
    height: 22px;
    background: var(--md-border);
    border-radius: 9999px;
    position: relative;
    transition: background 0.2s;
  }
  .switch-box::after {
    content: '';
    position: absolute;
    top: 3px;
    left: 3px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: white;
    transition: transform 0.2s;
  }
  input[type="checkbox"]:checked + .switch-box {
    background: var(--google-blue);
  }
  input[type="checkbox"]:checked + .switch-box::after {
    transform: translateX(16px);
  }

  /* Output Container */
  .output-box {
    margin-top: 20px;
    background: var(--md-surface);
    border: 1px solid var(--md-border);
    border-radius: 14px;
    overflow: hidden;
  }

  .output-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: var(--md-surface-variant);
    border-bottom: 1px solid var(--md-border);
    flex-wrap: wrap;
    gap: 10px;
  }

  .output-tabs {
    display: flex;
    gap: 4px;
  }

  .output-tab-btn {
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--md-text-secondary);
    background: transparent;
    border: none;
    cursor: pointer;
  }
  .output-tab-btn.active {
    background: var(--md-surface);
    color: var(--md-primary);
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
  }

  .output-body {
    padding: 18px;
    min-height: 140px;
    max-height: 480px;
    overflow-y: auto;
  }

  .output-text {
    font-size: 15px;
    line-height: 1.7;
    color: var(--md-text);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .output-code {
    font-family: "JetBrains Mono", monospace;
    font-size: 13px;
    line-height: 1.6;
    color: var(--md-text);
    white-space: pre-wrap;
    word-break: break-all;
  }

  .delta-text {
    color: var(--google-blue);
    font-style: italic;
    background: rgba(26,115,232,0.08);
    padding: 2px 6px;
    border-radius: 4px;
    display: inline-block;
  }

  /* Subtitle Timeline View */
  .timeline-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .timeline-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 10px 14px;
    border-radius: 8px;
    background: var(--md-bg);
    border: 1px solid var(--md-border);
    transition: background 0.15s ease;
  }
  .timeline-item:hover {
    background: var(--md-surface-variant);
  }
  .timeline-time {
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    font-weight: 600;
    color: var(--google-blue);
    padding: 3px 8px;
    border-radius: 6px;
    background: var(--md-primary-container);
    white-space: nowrap;
  }
  .timeline-text {
    font-size: 14px;
    color: var(--md-text);
    flex: 1;
  }

  /* Live Mic UI */
  .mic-stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 30px 20px 20px;
    text-align: center;
  }

  .mic-button-wrapper {
    position: relative;
    width: 100px;
    height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 20px;
  }

  .mic-ring {
    position: absolute;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    border: 2px solid var(--google-blue);
    opacity: 0;
    pointer-events: none;
  }
  .mic-button-wrapper.recording .mic-ring {
    border-color: var(--google-red);
    animation: ringPulse 1.8s infinite ease-out;
  }

  @keyframes ringPulse {
    0% { transform: scale(1); opacity: 0.8; }
    100% { transform: scale(1.6); opacity: 0; }
  }

  .mic-btn-main {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: none;
    background: linear-gradient(135deg, var(--google-blue), #4285f4);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 4px 14px rgba(26,115,232,0.4);
    transition: all 0.25s ease;
  }
  .mic-btn-main:hover {
    transform: scale(1.05);
  }
  .mic-btn-main.recording {
    background: linear-gradient(135deg, var(--google-red), #d93025);
    box-shadow: 0 4px 16px rgba(234,67,53,0.5);
  }
  .mic-btn-main .icon {
    font-size: 36px;
  }

  /* Audio Canvas Visualizer */
  canvas#visualizer {
    width: 100%;
    max-width: 600px;
    height: 60px;
    border-radius: 8px;
    margin-bottom: 16px;
    background: var(--md-bg);
    border: 1px solid var(--md-border);
  }

  /* Stats Grid */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
  }

  .stat-card {
    background: var(--md-surface);
    border: 1px solid var(--md-border);
    border-radius: 12px;
    padding: 16px;
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .stat-icon {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    background: var(--md-primary-container);
    color: var(--md-on-primary-container);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .stat-val {
    font-family: 'Google Sans', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--md-text);
  }
  .stat-label {
    font-size: 12px;
    color: var(--md-text-secondary);
  }

  /* Developer Hub Card */
  .api-doc-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .api-card {
    background: var(--md-surface);
    border: 1px solid var(--md-border);
    border-radius: 14px;
    padding: 20px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease;
  }
  .api-card:hover {
    border-color: var(--md-primary);
    box-shadow: var(--md-shadow-2);
  }

  .code-viewer {
    background: var(--md-code-bg);
    border: 1px solid var(--md-border);
    border-radius: 10px;
    padding: 14px;
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    color: var(--md-text);
    overflow-x: auto;
    position: relative;
    margin-top: 10px;
  }

  /* Toast */
  #toast {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%) translateY(100px);
    background: #202124;
    color: #fff;
    padding: 10px 20px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 1000;
    opacity: 0;
    transition: all 0.25s ease;
    display: flex;
    align-items: center;
    gap: 8px;
    border: 1px solid #3c4043;
  }
  #toast.show {
    transform: translateX(-50%) translateY(0);
    opacity: 1;
  }

  /* Responsiveness */
  @media (max-width: 640px) {
    header.google-app-bar { padding: 0 14px; }
    main { padding: 16px 12px 60px; }
    .brand-title { font-size: 16px; }
    .btn { padding: 8px 14px; font-size: 13px; }
  }
</style>
</head>

<body>
<!-- Top App Bar -->
<header class="google-app-bar">
  <a href="/" class="brand-group">
    <div class="logo-box">
      <span class="icon" style="font-size:24px;">graphic_eq</span>
    </div>
    <div>
      <div class="brand-title">
        QUAN-ASR Server <span class="brand-tag">EmbeddingGemma 2 · CTC</span>
      </div>
    </div>
  </a>

  <div class="header-actions">
    <!-- Server Status Chip -->
    <div class="badge status-loading" id="healthPill" title="Trạng thái mô hình và thiết bị tính toán">
      <span class="pulse-dot"></span>
      <span id="healthText">Đang tải...</span>
    </div>

    <!-- API Docs Links -->
    <a href="/scalar" target="_blank" class="nav-btn" title="Mở tài liệu API giao diện Scalar hiện đại">
      <span class="icon">menu_book</span> Scalar Docs
    </a>
    <a href="/docs" target="_blank" class="nav-btn" title="Mở Swagger UI (OpenAPI tiêu chuẩn)">
      <span class="icon">api</span> Swagger
    </a>

    <!-- Theme Toggle -->
    <button class="icon-btn" id="themeToggle" title="Chuyển chế độ sáng/tối">
      <span class="icon" id="themeIcon">light_mode</span>
    </button>
  </div>
</header>

<main>
  <!-- Model Loading Banner -->
  <div class="model-banner-container" id="modelLoadingBanner" style="display:none;">
    <div class="model-banner-card banner-card-loading">
      <div class="banner-icon-col">
        <div class="banner-spinner"></div>
      </div>
      <div class="banner-body">
        <div class="banner-headline">
          <span class="banner-title">Mô hình AI đang khởi tạo trọng số...</span>
          <span class="badge status-loading" id="bannerProgressBadge">Đang tải weights (~1.4 GB)</span>
        </div>
        <div class="banner-description" id="bannerDescText">
          Hệ thống đang tải và giải mã trọng số mô hình <strong>QUAN-Transcribe-1.0</strong> cùng Google Gemma Audio Encoder. Các tính năng nhận diện tệp, phát trực tiếp SSE và Microphone tạm thời được khóa để tránh quá tải. Trang demo sẽ tự động sẵn sàng ngay khi nạp xong!
        </div>
        <div class="banner-progress-track">
          <div class="banner-progress-bar"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Model Error Banner -->
  <div class="model-banner-container" id="modelErrorBanner" style="display:none;">
    <div class="model-banner-card banner-card-error">
      <div class="banner-icon-col">
        <span class="icon" style="font-size:28px; color:var(--google-red);">error</span>
      </div>
      <div class="banner-body">
        <div class="banner-headline">
          <span class="banner-title" style="color:var(--google-red);">Không thể nạp mô hình ASR</span>
          <span class="badge status-error">Lỗi khởi động</span>
        </div>
        <div class="banner-description" id="bannerErrorDetails">
          Đã xảy ra lỗi trong quá trình nạp trọng số mô hình. Vui lòng kiểm tra kết nối mạng đến Hugging Face hoặc biến môi trường HF_TOKEN.
        </div>
      </div>
      <button class="btn btn-outlined" onclick="refreshHealth()" style="align-self:center;">
        <span class="icon">refresh</span> Thử lại
      </button>
    </div>
  </div>

  <!-- Google Standard Navigation Tabs -->
  <nav class="google-tabs" role="tablist">
    <button class="tab-item active" data-tab="fileTab">
      <span class="icon">folder_open</span> Nhận diện Tệp âm thanh
    </button>
    <button class="tab-item" data-tab="micTab">
      <span class="icon">mic</span> Micro Trực tiếp (WebSocket)
    </button>
    <button class="tab-item" data-tab="metricsTab">
      <span class="icon">monitoring</span> Chỉ số & Hiệu năng
    </button>
    <button class="tab-item" data-tab="apiTab">
      <span class="icon">code</span> Tài liệu & Hướng dẫn Tích hợp
    </button>
  </nav>

  <!-- ================= TAB 1: FILE TRANSCRIPTION ================= -->
  <section id="fileTab" class="tab-content active">
    <!-- Sample Audio Picker -->
    <div class="samples-banner">
      <span class="samples-label"><span class="icon" style="font-size:16px;">library_music</span> Thử nhanh tệp mẫu:</span>
      <div id="sampleChipsContainer" style="display:flex; gap:8px; flex-wrap:wrap;">
        <button class="sample-chip" data-file="voicemail-90s.mp3">
          <span class="icon" style="font-size:16px;">play_circle</span> Voicemail nhà mạng (90s)
        </button>
        <button class="sample-chip" data-file="sales-call-72s.mp3">
          <span class="icon" style="font-size:16px;">play_circle</span> Cuộc gọi tư vấn (73s)
        </button>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <div>
          <h2 class="card-title"><span class="icon">upload_file</span> Tải lên hoặc chọn tệp âm thanh</h2>
          <p class="card-subtitle">Hỗ trợ MP3, WAV, M4A, FLAC, OGG, WebM · Tối đa 100MB</p>
        </div>
      </div>

      <!-- Drag & Drop Zone -->
      <div class="dropzone" id="dropzone">
        <input type="file" id="fileInput" accept="audio/*" style="display:none;" />
        <div class="dropzone-icon"><span class="icon" style="font-size:44px;">cloud_upload</span></div>
        <div class="dropzone-text">Kéo thả tệp âm thanh vào đây hoặc <span style="color:var(--google-blue); text-decoration:underline;">chọn từ máy tính</span></div>
        <div class="dropzone-sub">Xử lý tức thì với kiến trúc Single-Pass CTC siêu tốc</div>
      </div>

      <!-- Audio Preview Player -->
      <div class="audio-preview-box" id="audioPreviewBox">
        <audio id="audioPlayer" controls></audio>
        <div class="file-meta-pill">
          <span class="icon" style="font-size:18px;">audio_file</span>
          <span id="selectedFileName">chưa có tệp</span>
        </div>
        <div class="file-meta-pill" id="selectedFileSize">0 MB</div>
      </div>

      <!-- Controls Form -->
      <div class="form-grid">
        <div class="field-group">
          <label class="field-label" for="formatSelect">Định dạng phản hồi (Format)</label>
          <select id="formatSelect" class="google-input">
            <option value="json">json (OpenAI chuẩn: {"text": "..."})</option>
            <option value="verbose_json">verbose_json (Chi tiết mốc thời gian start / end)</option>
            <option value="text">text (Văn bản thuần túy không định dạng)</option>
            <option value="srt">srt (Tệp phụ đề video chuẩn SubRip)</option>
            <option value="vtt">vtt (Tệp phụ đề chuẩn WebVTT)</option>
          </select>
        </div>

        <div class="field-group">
          <label class="field-label" for="stepInput">Độ dài chunk stream (giây)</label>
          <input type="number" id="stepInput" class="google-input" value="0.48" step="0.08" min="0.08" max="5.0" />
        </div>
      </div>

      <!-- Advanced Settings Accordion -->
      <button class="accordion-toggle" id="advancedToggle">
        <span class="icon" id="advancedIcon">expand_more</span> Tùy chọn mô hình chuyên sâu (VAD, Context, KenLM)
      </button>

      <div class="accordion-content" id="advancedBox">
        <div class="form-grid" style="margin-bottom: 0;">
          <div class="field-group">
            <label class="field-label" for="leftChunksInput">Left Chunks (Ngữ cảnh quá khứ)</label>
            <input type="number" id="leftChunksInput" class="google-input" placeholder="Mặc định: server config" min="0" max="16" />
          </div>
          <div class="field-group">
            <label class="field-label" for="chunkSecondsInput">Chunk Seconds (Độ dài đoạn xử lý)</label>
            <input type="number" id="chunkSecondsInput" class="google-input" placeholder="Mặc định: 30s" min="1" max="60" />
          </div>
          <div class="field-group" style="justify-content: center;">
            <label class="switch-label">
              <input type="checkbox" id="streamPartialsCheck" checked style="display:none;" />
              <div class="switch-box"></div>
              <span>Hiển thị dự đoán thời gian thực (Stream Partials)</span>
            </label>
          </div>
          <div class="field-group" style="justify-content: center;">
            <label class="switch-label">
              <input type="checkbox" id="useLmCheck" style="display:none;" />
              <div class="switch-box"></div>
              <span>Sử dụng KenLM Beam Search (Language Model)</span>
            </label>
          </div>
          <div class="field-group" style="justify-content: center;">
            <label class="switch-label">
              <input type="checkbox" id="punctuateCheck" checked style="display:none;" />
              <div class="switch-box"></div>
              <span>Phục hồi dấu câu & Viết hoa (Punctuation + ITN)</span>
            </label>
          </div>
          <div class="field-group" style="justify-content: center;">
            <label class="switch-label">
              <input type="checkbox" id="seaG2pCheck" checked style="display:none;" />
              <div class="switch-box"></div>
              <span>Ký âm âm vị IPA (sea-g2p Phonemizer)</span>
            </label>
          </div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="actions-row">
        <button class="btn btn-primary" id="btnTranscribe" disabled>
          <span class="icon">bolt</span> Nhận diện nhanh (Batch Transcribe)
        </button>
        <button class="btn btn-tonal" id="btnStream" disabled>
          <span class="icon">stream</span> Truyền phát SSE (Stream it)
        </button>
        <button class="btn btn-outlined" id="btnClearFile">
          <span class="icon">clear</span> Làm mới
        </button>
        <div id="fileStatusChip" class="badge" style="display:none;"></div>
      </div>

      <!-- Output Display Box -->
      <div class="output-box" id="fileOutputBox" style="display:none;">
        <div class="output-header">
          <div class="output-tabs">
            <button class="output-tab-btn active" data-view="textView">Văn bản</button>
            <button class="output-tab-btn" data-view="timelineView" id="timelineTabBtn">Dòng thời gian</button>
            <button class="output-tab-btn" data-view="rawView">Mã JSON/Phụ đề</button>
          </div>

          <div style="display:flex; align-items:center; gap:8px;">
            <span id="transcribeStatBadge" class="badge"></span>
            <button class="btn-outlined" style="padding:4px 10px; font-size:12px;" id="btnCopyResult" title="Sao chép vào clipboard">
              <span class="icon" style="font-size:16px;">content_copy</span> Sao chép
            </button>
            <button class="btn-outlined" style="padding:4px 10px; font-size:12px;" id="btnDownloadResult" title="Tải xuống tệp">
              <span class="icon" style="font-size:16px;">download</span> Tải xuống
            </button>
          </div>
        </div>

        <div class="output-body">
          <div id="textView" class="output-text">—</div>
          <div id="timelineView" class="timeline-list" style="display:none;"></div>
          <pre id="rawView" class="output-code" style="display:none;">—</pre>
        </div>
      </div>
    </div>
  </section>


  <!-- ================= TAB 2: LIVE MICROPHONE (WEBSOCKET) ================= -->
  <section id="micTab" class="tab-content">
    <div class="card">
      <div class="card-header">
        <div>
          <h2 class="card-title"><span class="icon">mic</span> Nhận diện trực tiếp qua Micro (WebSocket Streaming)</h2>
          <p class="card-subtitle">Endpoint: <code>/v1/audio/transcriptions/stream?format=f32</code> · 16 kHz Mono</p>
        </div>
        <div id="wsBadge" class="badge">Đã ngắt kết nối</div>
      </div>

      <div class="mic-stage">
        <!-- Big Animated Mic Button -->
        <div class="mic-button-wrapper" id="micWrapper">
          <div class="mic-ring"></div>
          <button class="mic-btn-main" id="micBtn" title="Bấm để bắt đầu ghi âm">
            <span class="icon" id="micIcon">mic</span>
          </button>
        </div>

        <!-- Live Audio Waveform Canvas -->
        <canvas id="visualizer" width="600" height="60"></canvas>

        <div style="font-size:14px; font-weight:500; margin-bottom: 8px;" id="micStatusText">
          Nhấn vào biểu tượng Micro để nói tiếng Việt
        </div>

        <!-- Live Stream Options -->
        <div style="display:flex; gap:16px; justify-content:center; margin-bottom: 12px; flex-wrap: wrap;">
          <label class="switch-label">
            <input type="checkbox" id="punctuateMicCheck" checked style="display:none;" />
            <div class="switch-box"></div>
            <span>Dấu câu & Viết hoa</span>
          </label>
          <label class="switch-label">
            <input type="checkbox" id="seaG2pMicCheck" checked style="display:none;" />
            <div class="switch-box"></div>
            <span>Phonemes (sea-g2p)</span>
          </label>
        </div>

        <!-- Mic Action Controls -->
        <div class="actions-row" style="justify-content:center;">
          <button class="btn btn-outlined" id="btnMicPing" disabled>
            <span class="icon">wifi_tethering</span> Gửi Ping
          </button>
          <button class="btn btn-outlined" id="btnMicClear">
            <span class="icon">delete_sweep</span> Xóa nội dung
          </button>
          <button class="btn btn-outlined" id="btnMicCopy" disabled>
            <span class="icon">content_copy</span> Sao chép
          </button>
          <button class="btn btn-outlined" id="btnMicDownload" disabled>
            <span class="icon">download</span> Tải xuống
          </button>
        </div>
      </div>

      <!-- Live Stream Output View -->
      <div class="output-box" style="margin-top:16px;">
        <div class="output-header">
          <span style="font-size:12px; font-weight:600; text-transform:uppercase; color:var(--md-text-secondary);">
            Kết quả nhận diện trực tiếp
          </span>
          <div style="display:flex; gap:12px; font-size:12px; color:var(--md-text-muted);">
            <span id="micDurationLabel">Thời gian: 00:00</span>
            <span id="micPacketsLabel">Gói tin: 0</span>
          </div>
        </div>
        <div class="output-body" style="min-height:160px;">
          <div id="micFinals" style="font-size:16px; line-height:1.8; color:var(--md-text);"></div>
          <div id="micDelta" class="delta-text" style="display:none; margin-top:4px;"></div>
          <div id="micPhonemes" style="font-family: var(--font-code); color: var(--color-primary); font-size: 13px; margin-top: 10px; line-height: 1.6; word-break: break-word;"></div>
        </div>
      </div>
    </div>
  </section>


  <!-- ================= TAB 3: SERVER HEALTH & METRICS ================= -->
  <section id="metricsTab" class="tab-content">
    <!-- Stat Counters -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon"><span class="icon">speed</span></div>
        <div>
          <div class="stat-val" id="metricRtf">—</div>
          <div class="stat-label">Real-Time Factor (RTF)</div>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon" style="background:rgba(52,168,83,0.15); color:var(--google-green);"><span class="icon">timelapse</span></div>
        <div>
          <div class="stat-val" id="metricTotalAudio">—</div>
          <div class="stat-label">Tổng thời lượng âm thanh đã xử lý</div>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon" style="background:rgba(251,188,4,0.15); color:var(--google-yellow);"><span class="icon">task_alt</span></div>
        <div>
          <div class="stat-val" id="metricRequests">—</div>
          <div class="stat-label">Yêu cầu đã xử lý (Thành công / Lỗi)</div>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon" style="background:rgba(234,67,53,0.15); color:var(--google-red);"><span class="icon">memory</span></div>
        <div>
          <div class="stat-val" id="metricDevice">—</div>
          <div class="stat-label">Thiết bị tăng tốc (Inference Device)</div>
        </div>
      </div>
    </div>

    <!-- Details Card -->
    <div class="card">
      <div class="card-header">
        <h2 class="card-title"><span class="icon">info</span> Thông tin mô hình & Máy chủ</h2>
        <button class="btn btn-outlined" id="btnRefreshMetrics" style="padding:6px 12px; font-size:12px;">
          <span class="icon">refresh</span> Làm mới chỉ số
        </button>
      </div>

      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; font-size:13px; margin-bottom:20px;">
        <div><strong>Mô hình phục vụ:</strong> <span id="mInfoName">QUAN-Transcribe-1.0</span></div>
        <div><strong>Kiến trúc ASR:</strong> EmbeddingGemma 2 + CTC (356M params)</div>
        <div><strong>Ngôn ngữ:</strong> Tiếng Việt (Vietnamese, code: vi)</div>
        <div><strong>Hỗ trợ KenLM:</strong> <span id="mInfoLm">—</span></div>
        <div><strong>Bộ nhớ đệm (Queue Backpressure):</strong> <span id="mInfoQueue">—</span></div>
        <div><strong>Phiên WebSocket đang hoạt động:</strong> <span id="mInfoStreams">—</span></div>
      </div>

      <h3 style="font-size:13px; font-weight:600; text-transform:uppercase; color:var(--md-text-secondary); margin-bottom:8px;">
        Dữ liệu gốc từ <code>GET /metrics</code>
      </h3>
      <pre id="rawMetricsJson" class="output-code" style="background:var(--md-code-bg); padding:12px; border-radius:8px; max-height:240px; overflow-y:auto;">Đang tải...</pre>
    </div>
  </section>


  <!-- ================= TAB 4: API & DEVELOPER GUIDE ================= -->
  <section id="apiTab" class="tab-content">
    <!-- Scalar & Swagger Callout Cards -->
    <div class="api-doc-grid">
      <div class="api-card">
        <div>
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span class="icon" style="color:var(--google-blue); font-size:26px;">menu_book</span>
            <h3 style="font-size:17px; font-weight:600;">Scalar API Documentation</h3>
          </div>
          <p style="font-size:13px; color:var(--md-text-secondary); line-height:1.6;">
            Giao diện tài liệu API thế hệ mới của Scalar: tương tác trực tiếp, tự sinh mã nguồn đa ngôn ngữ (Python, JS, cURL, Go), tìm kiếm tức thì và giao diện đẹp chuẩn Google Kepler.
          </p>
        </div>
        <div style="margin-top:16px;">
          <a href="/scalar" target="_blank" class="btn btn-primary" style="width:100%; text-decoration:none;">
            <span class="icon">open_in_new</span> Mở Scalar API Reference
          </a>
        </div>
      </div>

      <div class="api-card">
        <div>
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
            <span class="icon" style="color:var(--google-green); font-size:26px;">api</span>
            <h3 style="font-size:17px; font-weight:600;">FastAPI Swagger UI</h3>
          </div>
          <p style="font-size:13px; color:var(--md-text-secondary); line-height:1.6;">
            Giao diện OpenAPI Swagger tiêu chuẩn tích hợp sẵn của FastAPI tại <code>/docs</code>, cho phép gửi thử nghiệm các endpoint POST multipart và GET thông số nhanh chóng.
          </p>
        </div>
        <div style="margin-top:16px;">
          <a href="/docs" target="_blank" class="btn btn-tonal" style="width:100%; text-decoration:none;">
            <span class="icon">open_in_new</span> Mở Swagger UI (/docs)
          </a>
        </div>
      </div>
    </div>

    <!-- Integration Code Examples -->
    <div class="card">
      <div class="card-header">
        <h2 class="card-title"><span class="icon">integration_instructions</span> Mẫu mã tích hợp nhanh</h2>
      </div>

      <!-- Tab switchers for code snippets -->
      <div class="output-tabs" style="margin-bottom:12px;">
        <button class="output-tab-btn active" data-code="codePython">Python (OpenAI SDK)</button>
        <button class="output-tab-btn" data-code="codeCurl">cURL (Command-Line)</button>
        <button class="output-tab-btn" data-code="codeJs">JavaScript / SSE</button>
        <button class="output-tab-btn" data-code="codeWs">WebSocket (Python/JS)</button>
      </div>

      <!-- Python SDK Snippet -->
      <div id="codePython" class="code-snippet-box">
        <div class="code-viewer">from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"  # Hoặc API_KEY cấu hình trên máy chủ
)

# Nhận diện âm thanh tiếng Việt với OpenAI SDK
with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="quan-transcribe-1.0",
        file=audio_file,
        response_format="verbose_json"  # hoặc "text", "srt", "vtt", "json"
    )

print(transcript)</div>
      </div>

      <!-- cURL Snippet -->
      <div id="codeCurl" class="code-snippet-box" style="display:none;">
        <div class="code-viewer"># 1. Nhận diện tiêu chuẩn
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "response_format=json"

# 2. Truyền phát Server-Sent Events (SSE) thời gian thực
curl -N -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "stream=true" \
  -F "stream_partials=true"</div>
      </div>

      <!-- JavaScript Fetch Snippet -->
      <div id="codeJs" class="code-snippet-box" style="display:none;">
        <div class="code-viewer">const formData = new FormData();
formData.append("file", fileInput.files[0]);
formData.append("response_format", "json");

const response = await fetch("http://localhost:8000/v1/audio/transcriptions", {
  method: "POST",
  body: formData
});
const result = await response.json();
console.log("Kết quả:", result.text);</div>
      </div>

      <!-- WebSocket Snippet -->
      <div id="codeWs" class="code-snippet-box" style="display:none;">
        <div class="code-viewer">// Kết nối WebSocket ghi âm trực tiếp 16 kHz Mono Float32
const ws = new WebSocket("ws://localhost:8000/v1/audio/transcriptions/stream?format=f32");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "transcript.text.delta") {
    console.log("Đang nhận diện:", data.text);
  } else if (data.type === "transcript.text.segment") {
    console.log("Câu hoàn chỉnh:", data.text);
  }
};

// Gửi mảng mẫu âm thanh Float32Array qua binary
// ws.send(float32AudioBuffer);</div>
      </div>
    </div>
  </section>
</main>

<!-- Toast Notification -->
<div id="toast"><span class="icon">check_circle</span> <span id="toastMsg">Thành công</span></div>

<script>
// ================= UTILITIES & STATE =================
const $ = (id) => document.getElementById(id);
const RESAMPLE_RATE = 16000;
let currentFile = null;
let currentResultText = "";
let currentResultFormat = "json";

// Toast Alert
function showToast(message, isError = false) {
  const t = $('toast');
  $('toastMsg').textContent = message;
  t.style.background = isError ? 'var(--google-red)' : '#202124';
  t.className = 'show';
  setTimeout(() => { t.className = ''; }, 3200);
}

// Theme Management
const themeToggle = $('themeToggle');
const themeIcon = $('themeIcon');

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('quan_asr_theme', theme);
  themeIcon.textContent = theme === 'dark' ? 'light_mode' : 'dark_mode';
}

const savedTheme = localStorage.getItem('quan_asr_theme') ||
  (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
setTheme(savedTheme);

themeToggle.onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme');
  setTheme(cur === 'dark' ? 'light' : 'dark');
};

// Tabs Switching
document.querySelectorAll('.tab-item').forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll('.tab-item').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach((c) => c.classList.remove('active'));
    btn.classList.add('active');
    const target = $(btn.dataset.tab);
    if (target) target.classList.add('active');
  };
});

// Output Sub-Tabs
document.querySelectorAll('.output-tab-btn[data-view]').forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll('.output-tab-btn[data-view]').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    $('textView').style.display = btn.dataset.view === 'textView' ? 'block' : 'none';
    $('timelineView').style.display = btn.dataset.view === 'timelineView' ? 'flex' : 'none';
    $('rawView').style.display = btn.dataset.view === 'rawView' ? 'block' : 'none';
  };
});

// Code Snippet Sub-Tabs
document.querySelectorAll('.output-tab-btn[data-code]').forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll('.output-tab-btn[data-code]').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.code-snippet-box').forEach((box) => box.style.display = 'none');
    const target = $(btn.dataset.code);
    if (target) target.style.display = 'block';
  };
});

// Advanced Options Accordion
const advancedToggle = $('advancedToggle');
const advancedBox = $('advancedBox');
const advancedIcon = $('advancedIcon');
advancedToggle.onclick = () => {
  const isExpanded = advancedBox.classList.toggle('expanded');
  advancedIcon.textContent = isExpanded ? 'expand_less' : 'expand_more';
};

// ================= STATE & READINESS =================
let isModelReady = false;
let healthTimer = null;

function updateUiReadiness(ready, loading, error, device) {
  const pill = $('healthPill');
  const text = $('healthText');
  const loadBanner = $('modelLoadingBanner');
  const errBanner = $('modelErrorBanner');
  const errDetails = $('bannerErrorDetails');
  const micBtn = $('micBtn');
  const sampleChips = document.querySelectorAll('.sample-chip');

  if (ready) {
    if (!isModelReady) {
      isModelReady = true;
      showToast('Mô hình QUAN-Transcribe-1.0 đã sẵn sàng phục vụ!');
    }
    pill.className = 'badge status-ready';
    text.textContent = `Sẵn sàng (${device || 'GPU/MPS'})`;
    $('metricDevice').textContent = device || 'Đang xác định';

    if (loadBanner) loadBanner.style.display = 'none';
    if (errBanner) errBanner.style.display = 'none';

    // Enable buttons if audio file selected
    if (currentFile) {
      btnTranscribe.disabled = false;
      btnStream.disabled = false;
    }
    btnTranscribe.innerHTML = '<span class="icon">play_arrow</span> Bắt đầu nhận diện';
    btnStream.innerHTML = '<span class="icon">stream</span> Phát trực tiếp (SSE)';

    micBtn.classList.remove('disabled-loading');
    micBtn.title = 'Nhấn vào biểu tượng Micro để nói tiếng Việt trực tiếp';
    sampleChips.forEach(c => c.classList.remove('disabled-loading'));

  } else if (error) {
    isModelReady = false;
    pill.className = 'badge status-error';
    text.textContent = 'Lỗi mô hình';
    $('metricDevice').textContent = device || 'Lỗi';

    if (loadBanner) loadBanner.style.display = 'none';
    if (errBanner) {
      errBanner.style.display = 'block';
      if (errDetails) errDetails.textContent = error;
    }

    btnTranscribe.disabled = true;
    btnStream.disabled = true;
    btnTranscribe.innerHTML = '<span class="icon">error</span> Mô hình bị lỗi';
    btnStream.innerHTML = '<span class="icon">error</span> Mô hình bị lỗi';
    micBtn.classList.add('disabled-loading');
    micBtn.title = 'Mô hình gặp lỗi khởi tạo';
    sampleChips.forEach(c => c.classList.add('disabled-loading'));

  } else {
    // Loading state
    isModelReady = false;
    pill.className = 'badge status-loading';
    text.textContent = 'Đang nạp mô hình...';
    $('metricDevice').textContent = device || 'Đang tải weights';

    if (loadBanner) loadBanner.style.display = 'block';
    if (errBanner) errBanner.style.display = 'none';

    btnTranscribe.disabled = true;
    btnStream.disabled = true;
    btnTranscribe.innerHTML = '<span class="icon spin">sync</span> Đang nạp mô hình...';
    btnStream.innerHTML = '<span class="icon spin">sync</span> Đang nạp mô hình...';

    micBtn.classList.add('disabled-loading');
    micBtn.title = 'Mô hình đang tải weights (~1.4 GB), vui lòng đợi...';
    sampleChips.forEach(c => c.classList.add('disabled-loading'));
  }
}

// ================= HEALTH & METRICS POLLING =================
async function refreshHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const isReady = Boolean(data.ready !== undefined ? data.ready : data.loaded);
    const isLoading = Boolean(data.loading || data.status === 'loading' || (!data.loaded && !data.error));
    updateUiReadiness(isReady, isLoading, data.error, data.device);

    if (data.lm !== undefined) {
      const lmCheck = $('useLmCheck');
      if (lmCheck) {
        if (!data.lm) {
          lmCheck.checked = false;
          lmCheck.disabled = true;
          const labelSpan = lmCheck.closest('label')?.querySelector('span');
          if (labelSpan) labelSpan.textContent = 'Sử dụng KenLM Beam Search (Chưa kích hoạt USE_LM)';
        } else {
          lmCheck.disabled = false;
          const labelSpan = lmCheck.closest('label')?.querySelector('span');
          if (labelSpan) labelSpan.textContent = 'Sử dụng KenLM Beam Search (Language Model)';
        }
      }
    }

    clearTimeout(healthTimer);
    healthTimer = setTimeout(refreshHealth, isReady ? 5000 : 1500);
  } catch (err) {
    $('healthPill').className = 'badge status-error';
    $('healthText').textContent = 'Ngoại tuyến (Offline)';
    clearTimeout(healthTimer);
    healthTimer = setTimeout(refreshHealth, 3000);
  }
}
refreshHealth();

async function refreshMetrics() {
  try {
    const res = await fetch('/metrics');
    const data = await res.json();
    $('rawMetricsJson').textContent = JSON.stringify(data, null, 2);

    // 1. Realtime Factor (RTF)
    const rtf = (data.realtime_factor !== undefined && data.realtime_factor !== null)
      ? data.realtime_factor
      : (data.rtf !== undefined ? data.rtf : null);
    if (rtf !== null && rtf !== undefined && rtf > 0) {
      const speed = (1 / rtf).toFixed(1);
      $('metricRtf').textContent = `${rtf.toFixed(2)}x (${speed}x realtime)`;
    } else {
      $('metricRtf').textContent = '—';
    }

    // 2. Total Audio Seconds
    const totalSec = data.audio_seconds_total !== undefined
      ? data.audio_seconds_total
      : (data.total_audio_seconds !== undefined ? data.total_audio_seconds : 0);
    if (totalSec > 0) {
      const mins = (totalSec / 60).toFixed(1);
      $('metricTotalAudio').textContent = `${mins} phút (${totalSec.toFixed(1)}s)`;
    } else {
      $('metricTotalAudio').textContent = '0s';
    }

    // 3. Requests Total (Success / Failed)
    const reqTotal = data.requests_total !== undefined ? data.requests_total : (data.requests || 0);
    const reqOk = data.requests_success !== undefined ? data.requests_success : reqTotal;
    const reqFail = data.requests_failed || 0;
    if (reqTotal > 0) {
      $('metricRequests').textContent = `${reqTotal} (${reqOk} OK / ${reqFail} lỗi)`;
    } else {
      $('metricRequests').textContent = '0 yêu cầu';
    }

    // 4. Device
    if (data.device) {
      $('metricDevice').textContent = data.device;
    }

    // 5. Model Info
    if (data.model) {
      $('mInfoName').textContent = data.model;
    }
    const hasLm = data.lm !== undefined ? data.lm : data.supports_lm;
    $('mInfoLm').textContent = hasLm ? 'Có (KenLM Beam Search)' : 'Không (Greedy CTC)';

    // 6. Queue & Batches
    if (data.core) {
      const queued = data.core.queued_pieces !== undefined ? data.core.queued_pieces : 0;
      const maxQ = data.max_queue || 256;
      const jobs = data.core.jobs_run !== undefined ? data.core.jobs_run : 0;
      $('mInfoQueue').textContent = `${queued} / ${maxQ} đoạn (Đã chạy: ${jobs} jobs)`;
    } else {
      $('mInfoQueue').textContent = '0 đoạn';
    }

    // 7. Active Streams
    const streams = data.active_streams !== undefined ? data.active_streams : 0;
    const batches = data.core && data.core.batches_run !== undefined ? ` · ${data.core.batches_run} batches` : '';
    $('mInfoStreams').textContent = `${streams} kết nối${batches}`;

  } catch (err) {
    $('rawMetricsJson').textContent = 'Không thể tải chỉ số metrics: ' + err;
  }
}
$('btnRefreshMetrics').onclick = refreshMetrics;
refreshMetrics();
setInterval(refreshMetrics, 3000);

// ================= FILE UPLOAD & TRANSCRIPTION =================
const dropzone = $('dropzone');
const fileInput = $('fileInput');
const audioPlayer = $('audioPlayer');
const audioPreviewBox = $('audioPreviewBox');
const selectedFileName = $('selectedFileName');
const selectedFileSize = $('selectedFileSize');
const btnTranscribe = $('btnTranscribe');
const btnStream = $('btnStream');

function setFile(file) {
  currentFile = file;
  if (!file) {
    audioPreviewBox.className = 'audio-preview-box';
    btnTranscribe.disabled = true;
    btnStream.disabled = true;
    return;
  }
  selectedFileName.textContent = file.name;
  selectedFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  audioPlayer.src = URL.createObjectURL(file);
  audioPreviewBox.className = 'audio-preview-box has-file';
  if (isModelReady) {
    btnTranscribe.disabled = false;
    btnStream.disabled = false;
    showToast(`Đã chọn tệp: ${file.name}`);
  } else {
    btnTranscribe.disabled = true;
    btnStream.disabled = true;
    showToast(`Đã chọn tệp: ${file.name}. Nút nhận diện sẽ mở ngay khi mô hình nạp xong!`, true);
  }
}

dropzone.onclick = () => fileInput.click();
fileInput.onchange = () => {
  if (fileInput.files.length > 0) setFile(fileInput.files[0]);
};

dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('dragover'); };
dropzone.ondragleave = () => dropzone.classList.remove('dragover');
dropzone.ondrop = (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) setFile(e.dataTransfer.files[0]);
};

$('btnClearFile').onclick = () => {
  setFile(null);
  fileInput.value = '';
  $('fileOutputBox').style.display = 'none';
  $('fileStatusChip').style.display = 'none';
};

// 1-Click Sample Audio Loading
document.querySelectorAll('.sample-chip').forEach((chip) => {
  chip.onclick = async () => {
    const filename = chip.dataset.file;
    showToast(`Đang tải tệp mẫu ${filename}...`);
    try {
      const res = await fetch(`/samples/${filename}`);
      if (!res.ok) throw new Error('Không tìm thấy tệp mẫu');
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'audio/mpeg' });
      setFile(file);
    } catch (err) {
      showToast('Lỗi tải tệp mẫu: ' + err.message, true);
    }
  };
});

// Render Results Helper
function displayResult(text, format, elapsed = 0) {
  currentResultText = text;
  currentResultFormat = format;
  $('fileOutputBox').style.display = 'block';

  $('transcribeStatBadge').textContent = elapsed > 0 ? `Hoàn tất trong ${elapsed.toFixed(2)}s` : 'Hoàn tất';
  $('rawView').textContent = text;

  // Process timeline if format has timestamps
  const timelineView = $('timelineView');
  const timelineTabBtn = $('timelineTabBtn');
  timelineView.innerHTML = '';

  let plainText = text;

  if (format === 'json' || format === 'verbose_json') {
    try {
      const parsed = JSON.parse(text);
      plainText = parsed.text || '';
      if (parsed.phonemes) {
        plainText += '\n\n[Ký âm âm vị IPA (sea-g2p)]:\n' + parsed.phonemes;
      }
      if (parsed.segments && Array.isArray(parsed.segments)) {
        timelineTabBtn.style.display = 'inline-block';
        parsed.segments.forEach((seg) => {
          const item = document.createElement('div');
          item.className = 'timeline-item';
          const start = (seg.start || 0).toFixed(2);
          const end = (seg.end || 0).toFixed(2);
          const phHtml = seg.phonemes ? `<div style="font-family: var(--font-code); color: var(--color-primary); font-size: 13px; margin-top: 4px;">IPA: ${seg.phonemes}</div>` : '';
          item.innerHTML = `
            <div class="timeline-time">${start}s - ${end}s</div>
            <div class="timeline-text">${seg.text}</div>
            ${phHtml}
          `;
          timelineView.appendChild(item);
        });
      } else {
        timelineTabBtn.style.display = 'none';
      }
    } catch (e) {
      timelineTabBtn.style.display = 'none';
    }
  } else if (format === 'srt' || format === 'vtt') {
    timelineTabBtn.style.display = 'inline-block';
    const lines = text.split('\n');
    let currentTime = '';
    let currentSegment = '';
    for (let line of lines) {
      line = line.trim();
      if (line.includes('-->')) {
        currentTime = line;
      } else if (line && isNaN(line) && !line.startsWith('WEBVTT')) {
        currentSegment = line;
        if (currentTime) {
          const item = document.createElement('div');
          item.className = 'timeline-item';
          item.innerHTML = `
            <div class="timeline-time">${currentTime}</div>
            <div class="timeline-text">${currentSegment}</div>
          `;
          timelineView.appendChild(item);
          currentTime = '';
        }
      }
    }
  } else {
    timelineTabBtn.style.display = 'none';
  }

  $('textView').textContent = plainText || '(Không có nội dung nhận diện)';
  // Switch to Text view by default
  document.querySelector('.output-tab-btn[data-view="textView"]').click();
}

// 1. Batch Transcribe Action
btnTranscribe.onclick = async () => {
  if (!isModelReady) {
    showToast('Mô hình đang nạp trọng số (~1.4 GB), vui lòng đợi trong giây lát!', true);
    return;
  }
  if (!currentFile) return;
  const format = $('formatSelect').value;
  const formData = new FormData();
  formData.append('file', currentFile);
  formData.append('response_format', format);

  if ($('leftChunksInput').value) formData.append('left_chunks', $('leftChunksInput').value);
  if ($('chunkSecondsInput').value) formData.append('chunk_seconds', $('chunkSecondsInput').value);
  if ($('useLmCheck').checked) formData.append('use_lm', 'true');
  if ($('punctuateCheck') && $('punctuateCheck').checked) formData.append('punctuate', 'true');
  if ($('seaG2pCheck') && $('seaG2pCheck').checked) formData.append('phonemes', 'true');

  btnTranscribe.disabled = true;
  btnStream.disabled = true;
  const chip = $('fileStatusChip');
  chip.style.display = 'inline-flex';
  chip.className = 'badge status-loading';
  chip.innerHTML = '<span class="pulse-dot"></span> Đang nhận diện âm thanh...';

  const t0 = performance.now();
  try {
    const res = await fetch('/v1/audio/transcriptions', { method: 'POST', body: formData });
    const text = await res.text();
    const elapsed = (performance.now() - t0) / 1000;

    if (res.ok) {
      chip.className = 'badge status-ready';
      chip.textContent = `Thành công (${elapsed.toFixed(2)}s)`;
      displayResult(text, format, elapsed);
      showToast('Đã nhận diện thành công!');
      refreshMetrics();
    } else {
      chip.className = 'badge status-error';
      chip.textContent = `Lỗi ${res.status}`;
      displayResult(`Lỗi từ máy chủ (${res.status}):\n${text}`, 'text', elapsed);
      showToast(`Lỗi: ${res.status}`, true);
    }
  } catch (err) {
    chip.className = 'badge status-error';
    chip.textContent = 'Lỗi kết nối';
    displayResult('Yêu cầu thất bại: ' + err.message, 'text', 0);
    showToast('Lỗi mạng: ' + err.message, true);
  } finally {
    btnTranscribe.disabled = false;
    btnStream.disabled = false;
  }
};

// 2. Real-time SSE Stream Action
btnStream.onclick = async () => {
  if (!isModelReady) {
    showToast('Mô hình đang nạp trọng số (~1.4 GB), vui lòng đợi trong giây lát!', true);
    return;
  }
  if (!currentFile) return;
  const formData = new FormData();
  formData.append('file', currentFile);
  formData.append('stream', 'true');
  formData.append('stream_partials', $('streamPartialsCheck').checked ? 'true' : 'false');
  if ($('leftChunksInput').value) formData.append('left_chunks', $('leftChunksInput').value);
  if ($('chunkSecondsInput').value) formData.append('chunk_seconds', $('chunkSecondsInput').value);
  if ($('useLmCheck').checked) formData.append('use_lm', 'true');
  if ($('punctuateCheck') && $('punctuateCheck').checked) formData.append('punctuate', 'true');
  if ($('seaG2pCheck') && $('seaG2pCheck').checked) formData.append('phonemes', 'true');

  btnTranscribe.disabled = true;
  btnStream.disabled = true;

  const chip = $('fileStatusChip');
  chip.style.display = 'inline-flex';
  chip.className = 'badge status-loading';
  chip.innerHTML = '<span class="pulse-dot"></span> Đang truyền phát SSE trực tiếp...';

  $('fileOutputBox').style.display = 'block';
  document.querySelector('.output-tab-btn[data-view="textView"]').click();
  $('textView').textContent = '';
  $('rawView').textContent = '';

  const t0 = performance.now();
  const confirmedSegments = [];

  try {
    const res = await fetch('/v1/audio/transcriptions', { method: 'POST', body: formData });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Mã lỗi ${res.status}: ${errText}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop();

      for (const frame of frames) {
        const line = frame.split('\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        const payload = JSON.parse(line.slice(6));
        $('rawView').textContent += JSON.stringify(payload) + '\n';

        if (payload.type === 'transcript.text.delta') {
          $('textView').innerHTML = confirmedSegments.join(' ') +
            ` <span class="delta-text">${payload.text}</span>`;
        } else if (payload.type === 'transcript.text.segment') {
          if (payload.text) confirmedSegments.push(payload.text);
          $('textView').textContent = confirmedSegments.join(' ');
        } else if (payload.type === 'transcript.text.done' || payload.type === 'transcript.text.completed') {
          const finalFull = payload.text || confirmedSegments.join(' ');
          displayResult(finalFull, 'text', payload.elapsed || ((performance.now() - t0) / 1000));
          refreshMetrics();
        } else if (payload.type === 'error') {
          showToast('Lỗi SSE: ' + payload.error.message, true);
        }
      }
    }

    chip.className = 'badge status-ready';
    chip.textContent = `SSE Hoàn tất (${((performance.now() - t0)/1000).toFixed(2)}s)`;
  } catch (err) {
    chip.className = 'badge status-error';
    chip.textContent = 'Lỗi truyền phát';
    $('textView').textContent = 'Lỗi truyền phát SSE: ' + err.message;
    showToast(err.message, true);
  } finally {
    btnTranscribe.disabled = false;
    btnStream.disabled = false;
  }
};

// Copy & Download Result
$('btnCopyResult').onclick = () => {
  if (!currentResultText) return;
  navigator.clipboard.writeText(currentResultText).then(() => {
    showToast('Đã sao chép kết quả vào bộ nhớ tạm!');
  });
};

$('btnDownloadResult').onclick = () => {
  if (!currentResultText) return;
  const extMap = { json: 'json', verbose_json: 'json', text: 'txt', srt: 'srt', vtt: 'vtt' };
  const ext = extMap[currentResultFormat] || 'txt';
  const blob = new Blob([currentResultText], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `quan_asr_${Date.now()}.${ext}`;
  a.click();
  showToast(`Đã tải xuống tệp .${ext}`);
};


// ================= LIVE MICROPHONE (WEBSOCKET STREAMING) =================
let ws = null;
let audioContext = null;
let micSource = null;
let workletNode = null;
let micStreamMedia = null;
let isRecording = false;
let packetCount = 0;
let micTimer = null;
let micSeconds = 0;
const liveFinalSegments = [];

// Canvas Visualizer
const canvas = $('visualizer');
const canvasCtx = canvas.getContext('2d');
let analyser = null;
let animFrameId = null;

function drawVisualizer() {
  if (!analyser || !isRecording) {
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  const barWidth = (canvas.width / bufferLength) * 2.5;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    const barHeight = (dataArray[i] / 255) * canvas.height;
    canvasCtx.fillStyle = '#1a73e8';
    canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
    x += barWidth + 1;
  }
  animFrameId = requestAnimationFrame(drawVisualizer);
}

const micBtn = $('micBtn');
const micWrapper = $('micWrapper');
const micIcon = $('micIcon');
const wsBadge = $('wsBadge');
const micStatusText = $('micStatusText');

async function startMic() {
  try {
    micStreamMedia = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 64;

    await audioContext.audioWorklet.addModule(URL.createObjectURL(new Blob([`
      class AudioTap extends AudioWorkletProcessor {
        process(inputs) {
          const ch = inputs[0][0];
          if (ch) this.port.postMessage(ch.slice());
          return true;
        }
      }
      registerProcessor('audio-tap', AudioTap);
    `], { type: 'application/javascript' })));

    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const punctuate = $('punctuateMicCheck') ? $('punctuateMicCheck').checked : true;
    const seaG2p = $('seaG2pMicCheck') ? $('seaG2pMicCheck').checked : true;
    ws = new WebSocket(`${scheme}://${location.host}/v1/audio/transcriptions/stream?format=f32&punctuate=${punctuate}&phonemes=${seaG2p}`);

    ws.onopen = () => {
      isRecording = true;
      packetCount = 0;
      micSeconds = 0;
      liveFinalSegments.length = 0;
      $('micFinals').textContent = '';
      $('micDelta').style.display = 'none';
      $('micPhonemes').textContent = '';

      micWrapper.classList.add('recording');
      micBtn.classList.add('recording');
      micIcon.textContent = 'stop';
      wsBadge.className = 'badge status-ready';
      wsBadge.innerHTML = '<span class="pulse-dot"></span> Đang ghi âm trực tiếp';
      micStatusText.textContent = 'Đang lắng nghe... Hãy nói tiếng Việt';
      $('btnMicPing').disabled = false;
      $('btnMicCopy').disabled = false;
      $('btnMicDownload').disabled = false;

      // Start Timer
      micTimer = setInterval(() => {
        micSeconds++;
        const m = String(Math.floor(micSeconds / 60)).padStart(2, '0');
        const s = String(micSeconds % 60).padStart(2, '0');
        $('micDurationLabel').textContent = `Thời gian: ${m}:${s}`;
      }, 1000);

      // Connect Audio Graph
      micSource = audioContext.createMediaStreamSource(micStreamMedia);
      workletNode = new AudioWorkletNode(audioContext, 'audio-tap');
      micSource.connect(analyser);
      micSource.connect(workletNode);

      workletNode.port.onmessage = (e) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const input = e.data;
        const ratio = audioContext.sampleRate / RESAMPLE_RATE;
        const out = new Float32Array(Math.floor(input.length / ratio));
        for (let i = 0; i < out.length; i++) {
          const idx = i * ratio;
          const lo = Math.floor(idx), hi = Math.min(lo + 1, input.length - 1);
          out[i] = input[lo] + (input[hi] - input[lo]) * (idx - lo);
        }
        ws.send(out.buffer);
        packetCount++;
        $('micPacketsLabel').textContent = `Gói tin: ${packetCount}`;
      };

      drawVisualizer();
      showToast('Đã kết nối micro WebSocket!');
    };

    ws.onmessage = (ev) => {
      const event = JSON.parse(ev.data);
      if (event.type === 'transcript.text.delta') {
        $('micDelta').style.display = 'inline-block';
        $('micDelta').textContent = event.text;
        if (event.phonemes) $('micPhonemes').textContent = 'IPA (Live): ' + event.phonemes;
      } else if (event.type === 'transcript.text.segment') {
        if (event.text) liveFinalSegments.push(event.text);
        $('micFinals').textContent = liveFinalSegments.join(' ');
        $('micDelta').style.display = 'none';
        if (event.phonemes) $('micPhonemes').textContent = 'IPA: ' + event.phonemes;
      } else if (event.type === 'transcript.text.done' && event.text) {
        if (!liveFinalSegments.includes(event.text)) liveFinalSegments.push(event.text);
        $('micFinals').textContent = liveFinalSegments.join(' ');
        $('micDelta').style.display = 'none';
        if (event.phonemes) $('micPhonemes').textContent = 'IPA (Hoàn tất): ' + event.phonemes;
      } else if (event.type === 'pong') {
        showToast('WebSocket Pong: Kết nối hoạt động tốt');
      } else if (event.type === 'error') {
        showToast('Lỗi WebSocket: ' + event.error.message, true);
      }
    };

    ws.onclose = () => {
      stopMic();
      wsBadge.className = 'badge';
      wsBadge.textContent = 'Đã ngắt kết nối';
    };

    ws.onerror = (e) => {
      showToast('Lỗi kết nối WebSocket', true);
      stopMic();
    };

  } catch (err) {
    showToast('Lỗi truy cập micro: ' + err.message, true);
    stopMic();
  }
}

function stopMic() {
  isRecording = false;
  if (micTimer) { clearInterval(micTimer); micTimer = null; }
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send('stop'); } catch (e) {}
    setTimeout(() => { ws.close(); }, 300);
  }

  workletNode?.disconnect();
  micSource?.disconnect();
  micStreamMedia?.getTracks().forEach((t) => t.stop());
  audioContext?.close();

  micWrapper.classList.remove('recording');
  micBtn.classList.remove('recording');
  micIcon.textContent = 'mic';
  micStatusText.textContent = 'Nhấn vào biểu tượng Micro để tiếp tục nói tiếng Việt';
  $('btnMicPing').disabled = true;
}

micBtn.onclick = () => {
  if (!isModelReady) {
    showToast('Mô hình đang nạp trọng số (~1.4 GB), vui lòng đợi trong giây lát!', true);
    return;
  }
  if (!isRecording) startMic();
  else stopMic();
};

$('btnMicPing').onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send('ping');
    showToast('Đã gửi ping...');
  }
};

$('btnMicClear').onclick = () => {
  liveFinalSegments.length = 0;
  $('micFinals').textContent = '';
  $('micDelta').style.display = 'none';
  $('micPhonemes').textContent = '';
  $('micDurationLabel').textContent = 'Thời gian: 00:00';
  $('micPacketsLabel').textContent = 'Gói tin: 0';
  showToast('Đã xóa nội dung nhận diện!');
};

$('btnMicCopy').onclick = () => {
  const text = liveFinalSegments.join(' ');
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => showToast('Đã sao chép nội dung micro!'));
};

$('btnMicDownload').onclick = () => {
  const text = liveFinalSegments.join(' ');
  if (!text) return;
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `quan_asr_mic_${Date.now()}.txt`;
  a.click();
  showToast('Đã tải xuống nội dung ghi âm!');
};
</script>
</body>
</html>
"""
