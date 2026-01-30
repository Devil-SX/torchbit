// Language switch functionality for torchbit documentation
(function() {
    // Get current path
    const path = window.location.pathname;

    // Determine if we should show English or Chinese based on path
    const isZh = path.includes('/zh/') || path.includes('/zh.');

    // Only redirect on the root page or if the language doesn't match the path
    if (path === '/' || path === '/index.html') {
        const userLang = navigator.language || navigator.userLanguage;
        if (userLang.startsWith('zh')) {
            window.location.href = '/zh/';
        }
    }
})();
