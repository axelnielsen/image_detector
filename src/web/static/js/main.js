/**
 * Script principal para la página de inicio del Detector de Patrones Oscuros
 * Gestiona la carga de archivos y el análisis directo de URLs
 */

document.addEventListener('DOMContentLoaded', function() {
    // Referencias a elementos del DOM
    const fileInput = document.getElementById('file-input');
    const fileName = document.getElementById('file-name');
    const uploadForm = document.getElementById('upload-form');
    const directForm = document.getElementById('direct-form');
    const directUrls = document.getElementById('direct-urls');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');

    // Actualizar nombre del archivo seleccionado
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            fileName.textContent = this.files[0].name;
        } else {
            fileName.textContent = 'Ningún archivo seleccionado';
        }
    });

    // Manejar envío del formulario de carga de archivo
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!fileInput.files || !fileInput.files[0]) {
            alert('Por favor, selecciona un archivo primero.');
            return;
        }

        // Mostrar overlay de carga
        loadingMessage.textContent = 'Cargando archivo y preparando análisis...';
        loadingOverlay.classList.remove('hidden');

        // Crear FormData y enviar
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Redirigir a la página de estado de la tarea
            window.location.href = data.redirect;
        })
        .catch(error => {
            loadingOverlay.classList.add('hidden');
            alert('Error: ' + error.message);
        });
    });

    // Manejar envío del formulario de análisis directo
    directForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const urlsText = directUrls.value.trim();
        if (!urlsText) {
            alert('Por favor, ingresa al menos una URL para analizar.');
            return;
        }

        // Extraer URLs (una por línea)
        const urls = urlsText.split('\n')
            .map(url => url.trim())
            .filter(url => url.length > 0);

        if (urls.length === 0) {
            alert('No se encontraron URLs válidas.');
            return;
        }

        // Mostrar overlay de carga
        loadingMessage.textContent = 'Preparando análisis de URLs...';
        loadingOverlay.classList.remove('hidden');

        // Enviar solicitud
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ urls: urls })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Redirigir a la página de estado de la tarea
            window.location.href = data.redirect;
        })
        .catch(error => {
            loadingOverlay.classList.add('hidden');
            alert('Error: ' + error.message);
        });
    });
});
