/**
 * Script para la página de estado de tarea del Detector de Patrones Oscuros
 * Gestiona la actualización del estado de la tarea y la visualización de resultados
 */

document.addEventListener('DOMContentLoaded', function() {
    // Referencias a elementos del DOM
    const taskId = document.getElementById('task-id').textContent;
    const taskStatus = document.getElementById('task-status');
    const taskProgress = document.getElementById('task-progress');
    const progressBar = document.getElementById('progress-bar');
    const processedUrls = document.getElementById('processed-urls');
    const totalUrls = document.getElementById('total-urls');
    const elapsedTime = document.getElementById('elapsed-time');
    
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');
    const errorText = document.getElementById('error-text');
    
    const totalSuccess = document.getElementById('total-success');
    const totalFailed = document.getElementById('total-failed');
    const totalDetections = document.getElementById('total-detections');
    
    const resultsTableBody = document.getElementById('results-table-body');
    
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    
    const exportSummaryCsv = document.getElementById('export-summary-csv');
    const exportAllJson = document.getElementById('export-all-json');
    
    // Variables para el seguimiento del tiempo
    let startTime = null;
    let timerInterval = null;
    
    // Iniciar verificación periódica del estado
    checkTaskStatus();
    const statusInterval = setInterval(checkTaskStatus, 3000);
    
    // Función para verificar el estado de la tarea
    function checkTaskStatus() {
        fetch(`/api/task/${taskId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error al obtener el estado de la tarea');
                }
                return response.json();
            })
            .then(data => {
                updateTaskStatus(data);
                
                // Si la tarea está completada o ha fallado, detener la verificación periódica
                if (data.status === 'completed' || data.status === 'failed') {
                    clearInterval(statusInterval);
                    
                    if (data.status === 'completed') {
                        loadTaskResults();
                    } else {
                        showError(data.error || 'Error desconocido durante el análisis');
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showError(error.message);
                clearInterval(statusInterval);
            });
    }
    
    // Función para actualizar la interfaz con el estado de la tarea
    function updateTaskStatus(data) {
        // Actualizar estado
        let statusText = '';
        switch(data.status) {
            case 'pending':
                statusText = 'Pendiente';
                break;
            case 'running':
                statusText = 'En progreso';
                break;
            case 'completed':
                statusText = 'Completado';
                break;
            case 'failed':
                statusText = 'Error';
                break;
            default:
                statusText = data.status;
        }
        taskStatus.textContent = statusText;
        
        // Actualizar progreso
        const progress = Math.round(data.progress);
        taskProgress.textContent = `${progress}%`;
        progressBar.style.width = `${progress}%`;
        
        // Actualizar contadores
        processedUrls.textContent = data.processed_urls;
        totalUrls.textContent = data.total_urls;
        
        // Iniciar temporizador si es la primera vez que la tarea está en ejecución
        if (data.status === 'running' && !startTime && data.start_time) {
            startTime = new Date(data.start_time);
            startTimer();
        }
        
        // Detener temporizador si la tarea ha finalizado
        if ((data.status === 'completed' || data.status === 'failed') && timerInterval) {
            clearInterval(timerInterval);
            
            // Establecer tiempo final si está disponible
            if (data.end_time) {
                const endTime = new Date(data.end_time);
                const duration = Math.floor((endTime - new Date(data.start_time)) / 1000);
                updateElapsedTime(duration);
            }
        }
    }
    
    // Función para iniciar el temporizador
    function startTimer() {
        timerInterval = setInterval(() => {
            const now = new Date();
            const duration = Math.floor((now - startTime) / 1000);
            updateElapsedTime(duration);
        }, 1000);
    }
    
    // Función para actualizar el tiempo transcurrido
    function updateElapsedTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        elapsedTime.textContent = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    
    // Función para cargar los resultados de la tarea
    function loadTaskResults() {
        loadingMessage.textContent = 'Cargando resultados...';
        loadingOverlay.classList.remove('hidden');
        
        fetch(`/api/task/${taskId}/results`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error al obtener los resultados');
                }
                return response.json();
            })
            .then(data => {
                displayResults(data);
                loadingOverlay.classList.add('hidden');
            })
            .catch(error => {
                console.error('Error:', error);
                showError(error.message);
                loadingOverlay.classList.add('hidden');
            });
    }
    
    // Función para mostrar los resultados
    function displayResults(results) {
        // Mostrar sección de resultados
        resultsSection.classList.remove('hidden');
        
        // Calcular estadísticas
        let successCount = 0;
        let failedCount = 0;
        let detectionsCount = 0;
        
        // Limpiar tabla de resultados
        resultsTableBody.innerHTML = '';
        
        // Procesar cada URL
        for (const url in results) {
            const result = results[url];
            
            if (result.success) {
                successCount++;
                detectionsCount += result.detection_count || 0;
                
                // Crear fila para la URL
                const row = document.createElement('tr');
                
                // URL
                const urlCell = document.createElement('td');
                const urlLink = document.createElement('a');
                urlLink.href = url;
                urlLink.target = '_blank';
                urlLink.textContent = url.length > 50 ? url.substring(0, 50) + '...' : url;
                urlCell.appendChild(urlLink);
                row.appendChild(urlCell);
                
                // Título
                const titleCell = document.createElement('td');
                titleCell.textContent = result.title || 'Sin título';
                row.appendChild(titleCell);
                
                // Estado
                const statusCell = document.createElement('td');
                statusCell.innerHTML = '<span class="status-success">Éxito</span>';
                row.appendChild(statusCell);
                
                // Patrones
                const patternsCell = document.createElement('td');
                if (result.detection_count > 0) {
                    patternsCell.innerHTML = `<strong>${result.detection_count}</strong> (${result.pattern_types.join(', ')})`;
                } else {
                    patternsCell.textContent = 'Ninguno';
                }
                row.appendChild(patternsCell);
                
                // Acciones
                const actionsCell = document.createElement('td');
                
                // Botón para ver informe HTML
                if (result.reports && result.reports.html) {
                    const htmlButton = document.createElement('a');
                    htmlButton.href = `/download/${result.reports.html}`;
                    htmlButton.target = '_blank';
                    htmlButton.className = 'btn btn-primary btn-sm';
                    htmlButton.innerHTML = '<i class="fas fa-file-alt"></i> HTML';
                    htmlButton.style.marginRight = '5px';
                    actionsCell.appendChild(htmlButton);
                }
                
                // Botón para descargar JSON
                if (result.reports && result.reports.json) {
                    const jsonButton = document.createElement('a');
                    jsonButton.href = `/download/${result.reports.json}`;
                    jsonButton.className = 'btn btn-secondary btn-sm';
                    jsonButton.innerHTML = '<i class="fas fa-file-code"></i> JSON';
                    jsonButton.style.marginRight = '5px';
                    actionsCell.appendChild(jsonButton);
                }
                
                // Botón para descargar CSV
                if (result.reports && result.reports.csv) {
                    const csvButton = document.createElement('a');
                    csvButton.href = `/download/${result.reports.csv}`;
                    csvButton.className = 'btn btn-secondary btn-sm';
                    csvButton.innerHTML = '<i class="fas fa-file-csv"></i> CSV';
                    actionsCell.appendChild(csvButton);
                }
                
                row.appendChild(actionsCell);
                
                // Añadir fila a la tabla
                resultsTableBody.appendChild(row);
            } else {
                failedCount++;
                
                // Crear fila para la URL con error
                const row = document.createElement('tr');
                
                // URL
                const urlCell = document.createElement('td');
                const urlLink = document.createElement('a');
                urlLink.href = url;
                urlLink.target = '_blank';
                urlLink.textContent = url.length > 50 ? url.substring(0, 50) + '...' : url;
                urlCell.appendChild(urlLink);
                row.appendChild(urlCell);
                
                // Título (vacío)
                const titleCell = document.createElement('td');
                titleCell.textContent = '-';
                row.appendChild(titleCell);
                
                // Estado
                const statusCell = document.createElement('td');
                statusCell.innerHTML = '<span class="status-error">Error</span>';
                row.appendChild(statusCell);
                
                // Patrones (vacío)
                const patternsCell = document.createElement('td');
                patternsCell.textContent = '-';
                row.appendChild(patternsCell);
                
                // Acciones (mensaje de error)
                const actionsCell = document.createElement('td');
                const errorSpan = document.createElement('span');
                errorSpan.className = 'error-message';
                errorSpan.textContent = result.error || 'Error desconocido';
                actionsCell.appendChild(errorSpan);
                row.appendChild(actionsCell);
                
                // Añadir fila a la tabla
                resultsTableBody.appendChild(row);
            }
        }
        
        // Actualizar estadísticas
        totalSuccess.textContent = successCount;
        totalFailed.textContent = failedCount;
        totalDetections.textContent = detectionsCount;
    }
    
    // Función para mostrar errores
    function showError(message) {
        errorText.textContent = message;
        errorSection.classList.remove('hidden');
    }
    
    // Manejar botones de exportación
    if (exportSummaryCsv) {
        exportSummaryCsv.addEventListener('click', function() {
            // Esta funcionalidad se implementaría en el backend
            alert('Funcionalidad de exportación de resumen CSV pendiente de implementar');
        });
    }
    
    if (exportAllJson) {
        exportAllJson.addEventListener('click', function() {
            // Esta funcionalidad se implementaría en el backend
            alert('Funcionalidad de exportación de todos los informes JSON pendiente de implementar');
        });
    }
});
