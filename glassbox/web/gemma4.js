/* GlassBox Gemma 4 Integration — Explain, Diagnose, Compare & Export */
/* Made with 🔮 by Karan Ray & contributors */

const Gemma4 = (() => {
    // ── DOM references ──
    const explainBtn = document.getElementById('explain-btn');
    const diagnoseBtn = document.getElementById('diagnose-btn');
    const compareToggle = document.getElementById('compare-toggle');
    const exportBtn = document.getElementById('export-btn');
    const compareSection = document.getElementById('compare-section');
    const compareSmolText = document.getElementById('compare-smol-text');
    const compareGemmaText = document.getElementById('compare-gemma-text');
    const compareLatencyBadge = document.getElementById('compare-latency-badge');
    const compareDiffBar = document.getElementById('compare-diff-bar');

    const gemmaPanel = document.getElementById('gemma-panel');
    const gemmaClose = document.getElementById('gemma-panel-close');
    const gemmaTitle = document.getElementById('gemma-panel-title');
    const gemmaBody = document.getElementById('gemma-panel-body');
    const runSuggestionBtn = document.getElementById('run-suggestion-btn');

    let _lastExplainResult = null;
    let _lastDiagnoseResult = null;
    let _lastCompareResult = null;
    let _getSteps = null;
    let _getPrompt = null;
    let _onRunSuggestion = null;

    // ── Helpers ──
    function showPanel(title, html) {
        if (!gemmaPanel || !gemmaTitle || !gemmaBody) return;
        gemmaTitle.textContent = title;
        gemmaBody.innerHTML = html;
        gemmaPanel.hidden = false;
        gemmaPanel.scrollTop = 0;
    }

    function hidePanel() {
        if (gemmaPanel) gemmaPanel.hidden = true;
    }

    function setLoading(btn, loading, loadingText = '⏳ Analyzing…') {
        if (!btn) return;
        if (loading) {
            btn.dataset.origText = btn.textContent;
            btn.textContent = loadingText;
            btn.disabled = true;
        } else {
            btn.textContent = btn.dataset.origText || btn.textContent;
            btn.disabled = false;
        }
    }

    function escHtml(s) {
        if (s == null) return '';
        const d = document.createElement('div');
        d.textContent = String(s);
        return d.innerHTML;
    }

    function buildTracePayload() {
        const steps = _getSteps ? _getSteps() : [];
        const prompt = _getPrompt ? _getPrompt() : '';

        const generated_tokens = steps.map((s, i) => ({
            step: i,
            token_str: s.token_str || '',
            token_id: s.token_id || 0,
        }));

        const entropy_per_step = steps.map(s => s.entropy_bits || 0);

        const top_k_per_step = steps.map(s => {
            const ids = s.top_logit_ids || [];
            const vals = s.top_logit_values || [];
            const strs = s.top_logit_strs || [];
            return ids.map((id, j) => ({
                token_str: strs[j] || '',
                prob: vals[j] || 0,
            }));
        });

        return { prompt, generated_tokens, entropy_per_step, top_k_per_step };
    }

    function getHeatmapB64() {
        const canvas = document.getElementById('attn-canvas');
        if (!canvas) return null;
        try {
            return canvas.toDataURL('image/png');
        } catch (_) {
            return null;
        }
    }

    // ── Render Explain Result ──
    function renderExplainResult(result) {
        let html = `<div class="gemma-section">
            <div class="gemma-section-title">📋 Summary</div>
            <p>${escHtml(result.summary || 'No summary available.')}</p>
        </div>`;

        if (result.key_observations && result.key_observations.length) {
            html += `<div class="gemma-section">
                <div class="gemma-section-title">🔍 Key Observations</div>
                <ul>${result.key_observations.map(o => `<li>${escHtml(o)}</li>`).join('')}</ul>
            </div>`;
        }

        if (result.suspicious_steps && result.suspicious_steps.length) {
            html += `<div class="gemma-section">
                <div class="gemma-section-title">⚠️ Suspicious Steps (${result.suspicious_steps.length})</div>
                <div class="gemma-suspicious-list">
                    ${result.suspicious_steps.map(s =>
                        `<div class="gemma-suspicious-item" data-step="${s.step_index}">
                            <span class="gemma-step-badge">Step ${s.step_index}</span>
                            <span>${escHtml(s.reason || '')}</span>
                        </div>`
                    ).join('')}
                </div>
            </div>`;
        } else {
            html += `<div class="gemma-section">
                <div class="gemma-section-title">✅ No Suspicious Steps</div>
                <p>Attention and entropy distributions appear stable across all generation steps.</p>
            </div>`;
        }

        return html;
    }

    // ── Render Diagnose Result ──
    function renderDiagnoseResult(result, originalPrompt) {
        let html = '';

        // Confidence score banner
        const score = typeof result.confidence_score === 'number' ? result.confidence_score : 70;
        let scoreClass = 'score-high';
        if (score < 50) scoreClass = 'score-low';
        else if (score < 75) scoreClass = 'score-mid';

        html += `<div class="gemma-section">
            <div class="gemma-section-title">
                <span>Model Confidence</span>
                <span class="gemma-score-badge ${scoreClass}">${score} / 100</span>
            </div>
            <div style="background:rgba(255,255,255,0.06); height:6px; border-radius:3px; overflow:hidden; margin-top:6px;">
                <div style="background:currentColor; width:${Math.min(100, Math.max(0, score))}%; height:100%; border-radius:3px;" class="${scoreClass}"></div>
            </div>
        </div>`;

        // Ambiguity critique
        if (result.ambiguity) {
            html += `<div class="gemma-section">
                <div class="gemma-section-title">🧐 Ambiguity Critique</div>
                <p>${escHtml(result.ambiguity)}</p>
            </div>`;
        }

        // Likely points of failure
        if (result.likely_points_of_failure && result.likely_points_of_failure.length) {
            html += `<div class="gemma-section">
                <div class="gemma-section-title">⚠️ Likely Points of Failure</div>
                <ul>${result.likely_points_of_failure.map(p => `<li>${escHtml(p)}</li>`).join('')}</ul>
            </div>`;
        }

        // Detected issues
        if (result.issues && result.issues.length) {
            html += `<div class="gemma-section">
                <div class="gemma-section-title">🩺 Trace Issues</div>
                <div class="gemma-suspicious-list">
                    ${result.issues.map(iss =>
                        `<div class="gemma-suspicious-item" style="border-color:rgba(139,92,246,0.3); background:rgba(139,92,246,0.1);">
                            <span class="gemma-step-badge" style="background:#7c3aed;">${escHtml(iss.type || 'issue')}</span>
                            <span>${escHtml(iss.description || '')}</span>
                        </div>`
                    ).join('')}
                </div>
            </div>`;
        }

        // Prompt Comparison & Suggested Rephrasing
        const suggested = result.suggested_rephrasing || result.suggested_prompt || originalPrompt;
        html += `<div class="gemma-section">
            <div class="gemma-section-title">💡 Suggested Rephrasing</div>
            <div class="gemma-prompt-compare">
                <div class="gemma-prompt-box gemma-prompt-original">
                    <div class="gemma-prompt-label">Original Prompt</div>
                    <div class="gemma-prompt-text">${escHtml(originalPrompt)}</div>
                </div>
                <div class="gemma-prompt-box gemma-prompt-suggested">
                    <div class="gemma-prompt-label">Improved Prompt</div>
                    <div class="gemma-prompt-text">${escHtml(suggested)}</div>
                </div>
            </div>
            ${result.explanation ? `<p style="font-size:0.8rem; color:#94a3b8; margin-top:6px;">${escHtml(result.explanation)}</p>` : ''}
        </div>`;

        return html;
    }

    function renderError(error, errorType) {
        let icon = '❌';
        let hint = '';
        if (errorType === 'missing_key') {
            icon = '🔑';
            hint = 'Add GEMINI_API_KEY to your .env file and restart the server.';
        } else if (errorType === 'rate_limit') {
            icon = '⏱️';
            hint = 'Gemini API rate limit reached. Wait a few seconds and try again.';
        }
        return `<div class="gemma-section" style="border-color:rgba(239,68,68,0.4); background:rgba(239,68,68,0.08);">
            <div class="gemma-section-title" style="color:#f87171;">${icon} Analysis Notice</div>
            <p style="color:#fca5a5;">${escHtml(error)}</p>
            ${hint ? `<p style="font-size:0.8rem; color:#fecaca; margin-top:6px;">💡 ${escHtml(hint)}</p>` : ''}
        </div>`;
    }

    // ── Highlight suspicious steps in token stream ──
    function highlightSuspicious(suspiciousSteps) {
        document.querySelectorAll('.chip-suspicious').forEach(el =>
            el.classList.remove('chip-suspicious')
        );
        if (!suspiciousSteps || !suspiciousSteps.length) return;

        const indices = new Set(suspiciousSteps.map(s => s.step_index));
        document.querySelectorAll('.chip-gen[data-step]').forEach(chip => {
            if (indices.has(Number(chip.dataset.step))) {
                chip.classList.add('chip-suspicious');
            }
        });
    }

    // ── Model Comparison Logic ──
    async function runModelComparison(prompt, smolText) {
        if (!compareSection || !compareToggle || !compareToggle.checked) return;

        compareSection.hidden = false;
        if (compareSmolText) compareSmolText.textContent = smolText || '(Generating...)';
        if (compareGemmaText) compareGemmaText.textContent = '⏳ Querying Gemma 4…';
        if (compareLatencyBadge) compareLatencyBadge.textContent = 'waiting…';

        try {
            const resp = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, max_tokens: 60, temperature: 0.7 }),
            });
            const data = await resp.json();

            if (data.error) {
                if (compareGemmaText) compareGemmaText.textContent = `❌ ${data.error}`;
                return;
            }

            _lastCompareResult = data;
            const gemmaText = data.completion || '';
            if (compareGemmaText) compareGemmaText.textContent = gemmaText;
            if (compareLatencyBadge) compareLatencyBadge.textContent = `${data.latency_ms || '—'} ms`;

            // Compute word-level divergence
            highlightComparisonDivergence(smolText, gemmaText);
        } catch (err) {
            if (compareGemmaText) compareGemmaText.textContent = `Network error: ${err.message}`;
        }
    }

    function highlightComparisonDivergence(smolText, gemmaText) {
        if (!smolText || !gemmaText || !compareDiffBar) return;
        const smolWords = smolText.trim().split(/\s+/);
        const gemmaWords = gemmaText.trim().split(/\s+/);

        let matchCount = 0;
        const maxLen = Math.max(smolWords.length, gemmaWords.length);
        for (let i = 0; i < maxLen; i++) {
            if (smolWords[i] && gemmaWords[i] && smolWords[i].toLowerCase() === gemmaWords[i].toLowerCase()) {
                matchCount++;
            }
        }
        const overlapPct = Math.round((matchCount / Math.max(maxLen, 1)) * 100);

        compareDiffBar.hidden = false;
        compareDiffBar.innerHTML = `<span style="font-size:0.75rem; color:#94a3b8;">Lexical overlap: <b>${overlapPct}%</b> · SmolLM2: ${smolWords.length} words · Gemma 4: ${gemmaWords.length} words</span>`;
    }

    // ── Export Interpretability Report ──
    function exportReport() {
        const steps = _getSteps ? _getSteps() : [];
        const prompt = _getPrompt ? _getPrompt() : '';
        const smolText = steps.map(s => s.token_str || '').join('');
        const heatmapImg = getHeatmapB64();

        const avgEntropy = steps.length
            ? (steps.reduce((a, s) => a + (s.entropy_bits || 0), 0) / steps.length).toFixed(2)
            : '0.00';
        const maxEntropy = steps.length
            ? Math.max(...steps.map(s => s.entropy_bits || 0)).toFixed(2)
            : '0.00';

        const now = new Date().toISOString().replace('T', ' ').slice(0, 19);

        let reportHtml = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>GlassBox Interpretability Report — ${escHtml(prompt.slice(0, 30))}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f121d; color: #e2e8f0; margin: 0; padding: 32px; line-height: 1.6; }
        .container { max-width: 900px; margin: 0 auto; background: #161b2c; border: 1px solid #2d3748; border-radius: 12px; padding: 32px; box-shadow: 0 16px 48px rgba(0,0,0,0.5); }
        h1 { color: #c4b5fd; margin-top: 0; font-size: 1.6rem; display: flex; align-items: center; gap: 8px; }
        .meta { font-size: 0.85rem; color: #94a3b8; margin-bottom: 24px; border-bottom: 1px solid #2d3748; padding-bottom: 12px; }
        .card { background: rgba(255,255,255,0.03); border: 1px solid #2d3748; border-radius: 8px; padding: 18px; margin-bottom: 20px; }
        .card h2 { margin-top: 0; font-size: 1.05rem; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.05em; }
        .prompt-box { background: rgba(0,0,0,0.3); border-left: 3px solid #7c3aed; padding: 12px 16px; border-radius: 4px; font-family: monospace; }
        .output-box { background: rgba(0,0,0,0.3); border-left: 3px solid #10b981; padding: 12px 16px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; }
        .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 10px; }
        .metric-cell { background: rgba(0,0,0,0.25); border: 1px solid #2d3748; border-radius: 6px; padding: 10px; text-align: center; }
        .metric-cell span { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; display: block; }
        .metric-cell b { font-size: 1.2rem; color: #e2e8f0; }
        table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.85rem; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #2d3748; }
        th { color: #a78bfa; font-weight: 600; }
        img.heatmap { max-width: 100%; border-radius: 8px; border: 1px solid #2d3748; margin-top: 10px; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
        .badge-suspicious { background: rgba(249,115,22,0.2); color: #fdba74; border: 1px solid rgba(249,115,22,0.5); }
        .footer { text-align: center; font-size: 0.78rem; color: #64748b; margin-top: 32px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 GlassBox Mechanistic Interpretability Report</h1>
        <div class="meta">
            <span>Generated on: <b>${now}</b></span> ·
            <span>Engine: <b>SmolLM2-135M (NumPy)</b></span> ·
            <span>Analysis: <b>Gemma 4 (Gemini API)</b></span> ·
            <span>Author: <b>Karan Ray</b></span>
        </div>

        <div class="card">
            <h2>User Prompt</h2>
            <div class="prompt-box">${escHtml(prompt)}</div>
        </div>

        <div class="card">
            <h2>SmolLM2-135M Generated Completion</h2>
            <div class="output-box">${escHtml(smolText || '(none)')}</div>
            <div class="metrics-grid">
                <div class="metric-cell"><span>Total Tokens</span><b>${steps.length}</b></div>
                <div class="metric-cell"><span>Avg Entropy</span><b>${avgEntropy} b</b></div>
                <div class="metric-cell"><span>Max Entropy</span><b>${maxEntropy} b</b></div>
                <div class="metric-cell"><span>KV Cache</span><b>Active</b></div>
            </div>
        </div>`;

        if (heatmapImg) {
            reportHtml += `<div class="card">
                <h2>Attention Matrix Heatmap Snapshot</h2>
                <img class="heatmap" src="${heatmapImg}" alt="Attention Heatmap">
            </div>`;
        }

        if (_lastExplainResult) {
            reportHtml += `<div class="card">
                <h2>Gemma 4 Trace Analysis</h2>
                <p><b>Summary:</b> ${escHtml(_lastExplainResult.summary || '')}</p>`;
            if (_lastExplainResult.key_observations && _lastExplainResult.key_observations.length) {
                reportHtml += `<ul>${_lastExplainResult.key_observations.map(o => `<li>${escHtml(o)}</li>`).join('')}</ul>`;
            }
            if (_lastExplainResult.suspicious_steps && _lastExplainResult.suspicious_steps.length) {
                reportHtml += `<h3>Suspicious Steps Identified:</h3><ul>`;
                _lastExplainResult.suspicious_steps.forEach(s => {
                    reportHtml += `<li><span class="badge badge-suspicious">Step ${s.step_index}</span> ${escHtml(s.reason)}</li>`;
                });
                reportHtml += `</ul>`;
            }
            reportHtml += `</div>`;
        }

        if (_lastDiagnoseResult) {
            reportHtml += `<div class="card">
                <h2>Gemma 4 Prompt & Hallucination Diagnostic</h2>
                <p><b>Estimated Confidence:</b> ${escHtml(_lastDiagnoseResult.confidence_score)} / 100</p>
                <p><b>Ambiguity:</b> ${escHtml(_lastDiagnoseResult.ambiguity || 'None')}</p>
                <p><b>Suggested Rephrasing:</b> ${escHtml(_lastDiagnoseResult.suggested_rephrasing || _lastDiagnoseResult.suggested_prompt || '')}</p>
            </div>`;
        }

        if (_lastCompareResult) {
            reportHtml += `<div class="card">
                <h2>Model Comparison: SmolLM2 vs Gemma 4</h2>
                <p><b>Gemma 4 Completion (${_lastCompareResult.latency_ms} ms):</b></p>
                <div class="output-box">${escHtml(_lastCompareResult.completion || '')}</div>
            </div>`;
        }

        reportHtml += `<div class="card">
            <h2>Per-Token Generation Trace (First 30 Steps)</h2>
            <table>
                <thead>
                    <tr><th>Step</th><th>Token</th><th>Entropy (bits)</th><th>Top Candidates</th></tr>
                </thead>
                <tbody>`;

        steps.slice(0, 30).forEach((s, i) => {
            const topStrs = (s.top_logit_strs || []).slice(0, 3).map((t, idx) => {
                const val = (s.top_logit_values || [])[idx];
                return `${t} (${(val || 0).toFixed(2)})`;
            }).join(', ');
            reportHtml += `<tr>
                <td>${i}</td>
                <td><code>${escHtml(s.token_str)}</code></td>
                <td>${(s.entropy_bits || 0).toFixed(2)}</td>
                <td>${escHtml(topStrs)}</td>
            </tr>`;
        });

        reportHtml += `</tbody></table></div>
        <div class="footer">Exported from GlassBox VisualToken Box · Mechanistic Interpretability Suite</div>
    </div></body></html>`;

        const blob = new Blob([reportHtml], { type: 'text/html;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `glassbox_report_${Date.now()}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ── Event Handlers ──
    if (explainBtn) {
        explainBtn.addEventListener('click', async () => {
            const steps = _getSteps ? _getSteps() : [];
            if (!steps.length) return;

            setLoading(explainBtn, true);
            const payload = buildTracePayload();
            payload.heatmap_b64_png = getHeatmapB64();

            try {
                const resp = await fetch('/api/explain', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await resp.json();

                if (data.error) {
                    showPanel('Explain Trace with Gemma 4', renderError(data.error, data.error_type));
                } else {
                    _lastExplainResult = data;
                    showPanel('Explain Trace with Gemma 4', renderExplainResult(data));
                    highlightSuspicious(data.suspicious_steps);

                    gemmaBody.querySelectorAll('.gemma-suspicious-item[data-step]').forEach(el => {
                        el.style.cursor = 'pointer';
                        el.addEventListener('click', () => {
                            const idx = Number(el.dataset.step);
                            const slider = document.getElementById('step-slider');
                            if (slider) {
                                slider.value = idx;
                                slider.dispatchEvent(new Event('input'));
                            }
                        });
                    });
                }
            } catch (err) {
                showPanel('Explain Trace with Gemma 4', renderError('Network error: ' + err.message, 'internal'));
            } finally {
                setLoading(explainBtn, false);
            }
        });
    }

    if (diagnoseBtn) {
        diagnoseBtn.addEventListener('click', async () => {
            const prompt = _getPrompt ? _getPrompt() : '';
            if (!prompt) return;

            setLoading(diagnoseBtn, true);
            const payload = buildTracePayload();

            try {
                const resp = await fetch('/api/diagnose', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await resp.json();

                if (data.error) {
                    showPanel('Diagnose Prompt', renderError(data.error, data.error_type));
                    if (runSuggestionBtn) runSuggestionBtn.hidden = true;
                } else {
                    _lastDiagnoseResult = data;
                    showPanel('Diagnose Prompt', renderDiagnoseResult(data, prompt));
                    const suggested = data.suggested_rephrasing || data.suggested_prompt;
                    if (runSuggestionBtn) {
                        runSuggestionBtn.hidden = !(suggested && suggested !== prompt);
                    }
                }
            } catch (err) {
                showPanel('Diagnose Prompt', renderError('Network error: ' + err.message, 'internal'));
                if (runSuggestionBtn) runSuggestionBtn.hidden = true;
            } finally {
                setLoading(diagnoseBtn, false);
            }
        });
    }

    if (compareToggle) {
        compareToggle.addEventListener('change', () => {
            if (compareSection) {
                compareSection.hidden = !compareToggle.checked;
            }
        });
    }

    if (exportBtn) {
        exportBtn.addEventListener('click', exportReport);
    }

    if (gemmaClose) {
        gemmaClose.addEventListener('click', () => {
            hidePanel();
            highlightSuspicious([]);
        });
    }

    if (runSuggestionBtn) {
        runSuggestionBtn.addEventListener('click', () => {
            const suggested = _lastDiagnoseResult && (_lastDiagnoseResult.suggested_rephrasing || _lastDiagnoseResult.suggested_prompt);
            if (suggested && _onRunSuggestion) {
                _onRunSuggestion(suggested);
                hidePanel();
            }
        });
    }

    // ── Public API ──
    return {
        setStepsGetter(fn) { _getSteps = fn; },
        setPromptGetter(fn) { _getPrompt = fn; },
        onRunSuggestion(fn) { _onRunSuggestion = fn; },
        setEnabled(enabled) {
            if (explainBtn) explainBtn.disabled = !enabled;
            if (diagnoseBtn) diagnoseBtn.disabled = !enabled;
        },
        onGenerationDone(steps, prompt) {
            if (explainBtn) explainBtn.disabled = false;
            if (diagnoseBtn) diagnoseBtn.disabled = false;
            const fullText = (steps || []).map(s => s.token_str || '').join('');
            if (compareToggle && compareToggle.checked) {
                runModelComparison(prompt, fullText);
            }
        },
        exportReport,
    };
})();
