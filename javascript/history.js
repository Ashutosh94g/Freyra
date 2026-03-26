/**
 * Freyra Generation History -- Browser-side helpers
 *
 * History is now stored server-side (modules/history_store.py) and rendered
 * via Gradio components. This file only provides global stubs so that any
 * legacy JS references don't throw errors.
 */

(function () {
    'use strict';

    window.freyraHistoryCapture = function () {
        // Server-side capture handles this now via the .then() chain
    };

    window.freyraHistoryExport = function () {};
    window.freyraHistoryClear = function () {};
    window.freyraHistoryImport = function () {};
})();
