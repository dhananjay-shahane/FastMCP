/**
 * MCP Data Analysis Server Dashboard JavaScript
 * Handles dashboard interactions, API calls, and dynamic content updates
 */

// Global variables
let systemStatus = {
    mcp_running: false,
    last_updated: null
};

let allFiles = [];
let fileInfoModal;
let executionModal;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap modals
    fileInfoModal = new bootstrap.Modal(document.getElementById('fileInfoModal'));
    executionModal = new bootstrap.Modal(document.getElementById('executionModal'));
    
    // Load initial data
    loadAllFiles();
    
    // Set up auto-refresh
    setInterval(refreshStatus, 30000); // Refresh every 30 seconds
    
    // Add event listeners
    setupEventListeners();
});

/**
 * Set up event listeners for various dashboard interactions
 */
function setupEventListeners() {
    // File upload drag and drop (if on upload page)
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        setupFileUpload();
    }
}

/**
 * Refresh system status
 */
async function refreshStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        if (data.success) {
            systemStatus = data.status;
            updateStatusDisplay();
        }
    } catch (error) {
        console.error('Error refreshing status:', error);
        showNotification('Failed to refresh system status', 'error');
    }
}

/**
 * Update status display elements
 */
function updateStatusDisplay() {
    // Update status indicator
    const statusIndicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.status-text');
    
    if (statusIndicator && statusText) {
        if (systemStatus.mcp_server_running) {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = 'Online';
        } else {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'Offline';
        }
    }
    
    // Update file statistics
    if (systemStatus.file_stats) {
        updateFileStats(systemStatus.file_stats);
    }
}

/**
 * Update file statistics display
 */
function updateFileStats(stats) {
    const csvCount = document.querySelector('.stat-number');
    if (csvCount) {
        const statNumbers = document.querySelectorAll('.stat-number');
        if (statNumbers.length >= 4) {
            statNumbers[1].textContent = stats.csv_files || 0;
            statNumbers[2].textContent = stats.script_files || 0;
            statNumbers[3].textContent = `${(stats.total_size / 1024).toFixed(1)} KB`;
        }
    }
}

/**
 * Load all files from the API
 */
async function loadAllFiles() {
    try {
        const response = await fetch('/api/files');
        const data = await response.json();
        
        if (data.success) {
            allFiles = data.files;
            displayAllFiles();
        } else {
            throw new Error(data.error || 'Failed to load files');
        }
    } catch (error) {
        console.error('Error loading files:', error);
        showFilesError('Failed to load files: ' + error.message);
    }
}

/**
 * Display all files in the files container
 */
function displayAllFiles() {
    const container = document.getElementById('files-container');
    if (!container) return;
    
    if (allFiles.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="fas fa-inbox fa-3x text-muted mb-3"></i>
                <p class="text-muted">No files available</p>
                <a href="/upload" class="btn btn-primary">
                    <i class="fas fa-upload me-1"></i>Upload Files
                </a>
            </div>
        `;
        return;
    }
    
    // Group files by type
    const csvFiles = allFiles.filter(f => f.type === 'csv');
    const scriptFiles = allFiles.filter(f => f.type === 'script');
    
    let html = '';
    
    // CSV Files Section
    if (csvFiles.length > 0) {
        html += `
            <h6 class="text-primary mb-3">
                <i class="fas fa-file-csv me-2"></i>
                CSV Data Files (${csvFiles.length})
            </h6>
            <div class="table-responsive mb-4">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Size</th>
                            <th>Modified</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        csvFiles.forEach(file => {
            const modifiedDate = new Date(file.modified).toLocaleDateString();
            html += `
                <tr>
                    <td>
                        <i class="fas fa-file-csv text-success me-2"></i>
                        ${file.name}
                    </td>
                    <td>${(file.size / 1024).toFixed(1)} KB</td>
                    <td>${modifiedDate}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary me-1" onclick="showFileInfo('${file.name}')">
                            <i class="fas fa-info-circle"></i>
                        </button>
                        <div class="btn-group">
                            <button class="btn btn-sm btn-outline-success dropdown-toggle" data-bs-toggle="dropdown">
                                <i class="fas fa-chart-bar"></i>
                            </button>
                            <ul class="dropdown-menu">
                                <li><a class="dropdown-item" href="#" onclick="createChart('${file.name}', 'bar_chart')">
                                    <i class="fas fa-chart-bar me-2"></i>Bar Chart
                                </a></li>
                                <li><a class="dropdown-item" href="#" onclick="createChart('${file.name}', 'line_graph')">
                                    <i class="fas fa-chart-line me-2"></i>Line Graph
                                </a></li>
                                <li><a class="dropdown-item" href="#" onclick="createChart('${file.name}', 'pie_chart')">
                                    <i class="fas fa-chart-pie me-2"></i>Pie Chart
                                </a></li>
                                <li><a class="dropdown-item" href="#" onclick="createChart('${file.name}', 'custom_stats')">
                                    <i class="fas fa-calculator me-2"></i>Statistics
                                </a></li>
                            </ul>
                        </div>
                    </td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
    }
    
    // Script Files Section
    if (scriptFiles.length > 0) {
        html += `
            <h6 class="text-warning mb-3">
                <i class="fas fa-file-code me-2"></i>
                Python Scripts (${scriptFiles.length})
            </h6>
            <div class="table-responsive">
                <table class="table table-hover">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Size</th>
                            <th>Modified</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        scriptFiles.forEach(file => {
            const modifiedDate = new Date(file.modified).toLocaleDateString();
            html += `
                <tr>
                    <td>
                        <i class="fas fa-file-code text-warning me-2"></i>
                        ${file.name}
                    </td>
                    <td>${(file.size / 1024).toFixed(1)} KB</td>
                    <td>${modifiedDate}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary me-1" onclick="showFileInfo('${file.name}')">
                            <i class="fas fa-info-circle"></i>
                        </button>
                        <button class="btn btn-sm btn-outline-success" onclick="executeScript('${file.name}')">
                            <i class="fas fa-play me-1"></i>Run
                        </button>
                    </td>
                </tr>
            `;
        });
        
        html += `
                    </tbody>
                </table>
            </div>
        `;
    }
    
    container.innerHTML = html;
}

/**
 * Show error message in files container
 */
function showFilesError(message) {
    const container = document.getElementById('files-container');
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
                <button class="btn btn-sm btn-outline-danger ms-2" onclick="loadAllFiles()">
                    <i class="fas fa-sync-alt"></i> Retry
                </button>
            </div>
        `;
    }
}

/**
 * Show detailed file information in modal
 */
async function showFileInfo(filename) {
    const content = document.getElementById('fileInfoContent');
    
    // Show loading state
    content.innerHTML = `
        <div class="text-center py-3">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Loading file information...</p>
        </div>
    `;
    
    fileInfoModal.show();
    
    try {
        const response = await fetch(`/api/file/${encodeURIComponent(filename)}`);
        const data = await response.json();
        
        if (data.success) {
            displayFileInfo(data.file_info);
        } else {
            throw new Error(data.error || 'Failed to load file information');
        }
    } catch (error) {
        console.error('Error loading file info:', error);
        content.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Failed to load file information: ${error.message}
            </div>
        `;
    }
}

/**
 * Display file information in modal
 */
function displayFileInfo(fileInfo) {
    const content = document.getElementById('fileInfoContent');
    let html = `
        <div class="row mb-3">
            <div class="col-md-6">
                <strong>Name:</strong> ${fileInfo.name}<br>
                <strong>Size:</strong> ${(fileInfo.size / 1024).toFixed(2)} KB<br>
                <strong>Type:</strong> ${fileInfo.type.toUpperCase()}
            </div>
            <div class="col-md-6">
                <strong>Modified:</strong> ${new Date(fileInfo.modified).toLocaleString()}
            </div>
        </div>
    `;
    
    if (fileInfo.type === 'csv') {
        html += `
            <div class="row mb-3">
                <div class="col-md-6">
                    <strong>Rows:</strong> ${fileInfo.rows}<br>
                    <strong>Columns:</strong> ${fileInfo.columns}
                </div>
            </div>
            
            <h6 class="text-primary">Column Names:</h6>
            <div class="mb-3">
                ${fileInfo.column_names.map(col => `<span class="badge bg-secondary me-1">${col}</span>`).join('')}
            </div>
            
            <h6 class="text-primary">Data Preview:</h6>
            <div class="table-responsive">
                <table class="table table-sm table-striped">
                    <thead>
                        <tr>
                            ${fileInfo.column_names.map(col => `<th>${col}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${fileInfo.preview.map(row => `
                            <tr>
                                ${fileInfo.column_names.map(col => `<td>${row[col] || ''}</td>`).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    } else if (fileInfo.type === 'script') {
        html += `
            <div class="row mb-3">
                <div class="col-md-6">
                    <strong>Lines:</strong> ${fileInfo.lines}
                </div>
            </div>
            
            <h6 class="text-primary">Script Preview:</h6>
            <div class="bg-light p-3" style="border-radius: 0.375rem;">
                <pre style="font-size: 0.875rem; margin: 0; white-space: pre-wrap;">${fileInfo.preview}</pre>
            </div>
        `;
    }
    
    content.innerHTML = html;
}

/**
 * Execute a script
 */
async function executeScript(scriptName, args = []) {
    const content = document.getElementById('executionContent');
    
    // Show loading state
    content.innerHTML = `
        <div class="text-center py-3">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Executing...</span>
            </div>
            <p class="mt-2">Executing ${scriptName}...</p>
            <small class="text-muted">This may take a few moments</small>
        </div>
    `;
    
    executionModal.show();
    
    try {
        const response = await fetch('/api/execute-script', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                script_name: scriptName,
                args: args
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            content.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    Script executed successfully!
                </div>
                <div class="bg-light p-3" style="border-radius: 0.375rem;">
                    <strong>Script:</strong> ${scriptName}<br>
                    <strong>Message:</strong> ${data.message}
                </div>
            `;
        } else {
            throw new Error(data.error || 'Script execution failed');
        }
    } catch (error) {
        console.error('Error executing script:', error);
        content.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Script execution failed: ${error.message}
            </div>
        `;
    }
}

/**
 * Create chart from file
 */
function createChart(filename, chartType) {
    // For now, execute the appropriate script
    const scriptMap = {
        'bar_chart': 'bar_chart.py',
        'line_graph': 'line_graph.py',
        'pie_chart': 'pie_chart.py',
        'custom_stats': 'custom_stats.py'
    };
    
    const scriptName = scriptMap[chartType];
    if (scriptName) {
        executeScript(scriptName, [filename]);
    }
}

/**
 * Show notification to user
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format date for display
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showNotification('Copied to clipboard!', 'success');
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        showNotification('Failed to copy to clipboard', 'error');
    }
}

/**
 * Download file (if API supports it)
 */
function downloadFile(filename) {
    const link = document.createElement('a');
    link.href = `/api/download/${encodeURIComponent(filename)}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * Refresh page data
 */
function refreshPage() {
    window.location.reload();
}

// Export functions for global access
window.refreshStatus = refreshStatus;
window.showFileInfo = showFileInfo;
window.executeScript = executeScript;
window.createChart = createChart;
window.showNotification = showNotification;
window.copyToClipboard = copyToClipboard;
window.downloadFile = downloadFile;
window.refreshPage = refreshPage;
