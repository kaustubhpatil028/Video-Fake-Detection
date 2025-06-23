class PopupManager {
    constructor() {
        this.initializeElements();
        this.initializeEventListeners();
    }

    initializeElements() {
        this.downloadBtn = document.querySelector('#download-button');
        this.urlInput = document.querySelector('#content-url');
        this.statusMessage = document.querySelector('#status-message');

        if (!this.downloadBtn || !this.urlInput || !this.statusMessage) {
            console.error('One or more required elements not found');
            return false;
        }
        return true;
    }

    showStatus(message, type = 'info') {
        if (!this.statusMessage) {
            console.error('Status message element not found');
            return;
        }
        this.statusMessage.textContent = message;
        this.statusMessage.className = type;
    }

    async downloadContent(url) {
        if (!url) {
            this.showStatus('Please enter a content URL', 'error');
            return;
        }

        try {
            this.showStatus('Processing content...');
            const response = await fetch('http://localhost:8000/download', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url })
            });

            const data = await response.json();
            
            if (response.ok) {
                this.showStatus('Download completed successfully!', 'success');
            } else {
                this.showStatus(data.detail || 'Download failed', 'error');
            }
        } catch (error) {
            this.showStatus('Failed to connect to the server', 'error');
            console.error('Error:', error);
        }
    }

    initializeEventListeners() {
        if (!this.downloadBtn) {
            console.error('Download button not found');
            return;
        }

        this.downloadBtn.addEventListener('click', () => {
            this.downloadContent(this.urlInput.value);
        });
    }
}

// Initialize the popup when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        new PopupManager();
    } catch (error) {
        console.error('Failed to initialize popup:', error);
    }
});
