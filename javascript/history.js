/**
 * Freyra Browser-Local Generation History
 *
 * Stores generation results in localStorage so they persist across page reloads.
 * Each entry captures: timestamp, seed, thumbnail, prompt preview, and dimension settings.
 * Capped at 200 entries (~5MB localStorage budget).
 */

(function () {
    'use strict';

    var STORAGE_KEY = 'freyra_history';
    var MAX_ENTRIES = 200;
    var THUMB_SIZE = 128;
    var THUMB_QUALITY = 0.6;

    function getHistory() {
        try {
            var raw = localStorage.getItem(STORAGE_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            return [];
        }
    }

    function saveHistory(history) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
        } catch (e) {
            if (history.length > 10) {
                history = history.slice(0, Math.floor(history.length / 2));
                try {
                    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
                } catch (e2) { /* give up */ }
            }
        }
    }

    function createThumbnail(imgElement) {
        return new Promise(function (resolve) {
            if (!imgElement || !imgElement.src) {
                resolve(null);
                return;
            }
            var img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = function () {
                var canvas = document.createElement('canvas');
                canvas.width = THUMB_SIZE;
                canvas.height = THUMB_SIZE;
                var ctx = canvas.getContext('2d');

                var scale = Math.max(THUMB_SIZE / img.width, THUMB_SIZE / img.height);
                var sw = THUMB_SIZE / scale, sh = THUMB_SIZE / scale;
                var sx = (img.width - sw) / 2, sy = (img.height - sh) / 2;

                ctx.drawImage(img, sx, sy, sw, sh, 0, 0, THUMB_SIZE, THUMB_SIZE);
                resolve(canvas.toDataURL('image/jpeg', THUMB_QUALITY));
            };
            img.onerror = function () { resolve(null); };
            img.src = imgElement.src;
        });
    }

    function getPromptPreview() {
        var el = document.querySelector('[data-testid="textbox"][aria-label="Assembled Prompt (read-only)"]');
        if (!el) {
            var labels = document.querySelectorAll('label');
            for (var i = 0; i < labels.length; i++) {
                if (labels[i].textContent.indexOf('Assembled Prompt') !== -1) {
                    var container = labels[i].closest('.gradio-textbox, .svelte-1gfkn6j');
                    if (container) {
                        el = container.querySelector('textarea, input');
                    }
                    break;
                }
            }
        }
        return el ? el.value || '' : '';
    }

    function getSeedDisplay() {
        var labels = document.querySelectorAll('label');
        for (var i = 0; i < labels.length; i++) {
            if (labels[i].textContent.indexOf('Last Seed') !== -1) {
                var container = labels[i].closest('.gradio-textbox, .svelte-1gfkn6j');
                if (container) {
                    var input = container.querySelector('textarea, input');
                    return input ? input.value || '' : '';
                }
            }
        }
        return '';
    }

    async function captureEntry() {
        var gallery = document.querySelector('.freyra-gallery');
        if (!gallery) return null;

        var images = gallery.querySelectorAll('img');
        if (images.length === 0) return null;

        var thumb = await createThumbnail(images[0]);
        var prompt = getPromptPreview();
        var seed = getSeedDisplay();

        return {
            id: Date.now() + '_' + Math.random().toString(36).substr(2, 6),
            timestamp: new Date().toISOString(),
            seed: seed,
            prompt: prompt.substring(0, 500),
            thumbnail: thumb,
            imageCount: images.length,
        };
    }

    function addEntry(entry) {
        if (!entry) return;
        var history = getHistory();
        history.unshift(entry);
        if (history.length > MAX_ENTRIES) {
            history = history.slice(0, MAX_ENTRIES);
        }
        saveHistory(history);
        renderHistoryPanel();
    }

    function clearHistory() {
        localStorage.removeItem(STORAGE_KEY);
        renderHistoryPanel();
    }

    function exportHistory() {
        var history = getHistory();
        var blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'freyra_history_' + new Date().toISOString().slice(0, 10) + '.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function importHistory(file) {
        var reader = new FileReader();
        reader.onload = function (e) {
            try {
                var imported = JSON.parse(e.target.result);
                if (Array.isArray(imported)) {
                    var history = getHistory();
                    var existingIds = {};
                    for (var i = 0; i < history.length; i++) {
                        existingIds[history[i].id] = true;
                    }
                    for (var j = 0; j < imported.length; j++) {
                        if (!existingIds[imported[j].id]) {
                            history.push(imported[j]);
                        }
                    }
                    history.sort(function (a, b) {
                        return new Date(b.timestamp) - new Date(a.timestamp);
                    });
                    if (history.length > MAX_ENTRIES) {
                        history = history.slice(0, MAX_ENTRIES);
                    }
                    saveHistory(history);
                    renderHistoryPanel();
                }
            } catch (err) {
                console.error('Failed to import history:', err);
            }
        };
        reader.readAsText(file);
    }

    function renderHistoryPanel() {
        var panel = document.getElementById('freyra-history-panel');
        if (!panel) return;

        var history = getHistory();
        var html = '<div class="freyra-history-controls">';
        html += '<span style="color:#888;font-size:13px;">' + history.length + ' generations saved</span>';
        html += '<div style="display:flex;gap:8px;">';
        html += '<button onclick="freyraHistoryExport()" class="freyra-hist-btn">Export</button>';
        html += '<label class="freyra-hist-btn" style="cursor:pointer;">Import<input type="file" accept=".json" onchange="freyraHistoryImport(event)" style="display:none;"></label>';
        html += '<button onclick="freyraHistoryClear()" class="freyra-hist-btn freyra-hist-btn-danger">Clear All</button>';
        html += '</div></div>';

        if (history.length === 0) {
            html += '<div style="text-align:center;padding:40px;color:#666;">No generation history yet. Generate some images to see them here.</div>';
        } else {
            html += '<div class="freyra-history-grid">';
            for (var i = 0; i < history.length; i++) {
                var entry = history[i];
                var date = new Date(entry.timestamp);
                var dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                var seedStr = entry.seed || '?';
                var promptShort = (entry.prompt || '').substring(0, 80);
                if ((entry.prompt || '').length > 80) promptShort += '...';

                html += '<div class="freyra-history-card" title="' + (entry.prompt || '').replace(/"/g, '&quot;') + '">';
                if (entry.thumbnail) {
                    html += '<img src="' + entry.thumbnail + '" class="freyra-history-thumb">';
                } else {
                    html += '<div class="freyra-history-thumb freyra-history-no-thumb">No preview</div>';
                }
                html += '<div class="freyra-history-meta">';
                html += '<div class="freyra-history-date">' + dateStr + '</div>';
                html += '<div class="freyra-history-seed">Seed: ' + seedStr + '</div>';
                html += '<div class="freyra-history-prompt">' + promptShort + '</div>';
                html += '<div class="freyra-history-count">' + (entry.imageCount || 1) + ' image(s)</div>';
                html += '</div></div>';
            }
            html += '</div>';
        }

        panel.innerHTML = html;
    }

    window.freyraHistoryExport = exportHistory;
    window.freyraHistoryClear = function () {
        if (confirm('Clear all generation history? This cannot be undone.')) {
            clearHistory();
        }
    };
    window.freyraHistoryImport = function (event) {
        var file = event.target.files[0];
        if (file) importHistory(file);
    };

    window.freyraHistoryCapture = async function () {
        var entry = await captureEntry();
        addEntry(entry);
    };

    var observer = new MutationObserver(function () {
        var panel = document.getElementById('freyra-history-panel');
        if (panel && !panel.dataset.rendered) {
            panel.dataset.rendered = 'true';
            renderHistoryPanel();
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    setTimeout(function () {
        renderHistoryPanel();
    }, 3000);
})();
