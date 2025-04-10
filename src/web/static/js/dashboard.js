/**
 * Script para la página de dashboard del Detector de Patrones Oscuros
 * Gestiona la visualización de estadísticas y gráficos
 */

document.addEventListener('DOMContentLoaded', function() {
    // Referencias a elementos del DOM
    const totalTasks = document.getElementById('total-tasks');
    const totalUrls = document.getElementById('total-urls');
    const totalDetections = document.getElementById('total-detections');
    const patternsChart = document.getElementById('patterns-chart');
    
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    
    // Cargar datos del dashboard
    loadDashboardData();
    
    // Función para cargar datos del dashboard
    function loadDashboardData() {
        loadingMessage.textContent = 'Cargando estadísticas...';
        loadingOverlay.classList.remove('hidden');
        
        fetch('/api/dashboard/summary')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error al obtener los datos del dashboard');
                }
                return response.json();
            })
            .then(data => {
                updateDashboard(data);
                loadingOverlay.classList.add('hidden');
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error al cargar los datos del dashboard: ' + error.message);
                loadingOverlay.classList.add('hidden');
            });
    }
    
    // Función para actualizar el dashboard con los datos recibidos
    function updateDashboard(data) {
        // Actualizar estadísticas
        totalTasks.textContent = data.total_tasks;
        totalUrls.textContent = data.total_urls;
        totalDetections.textContent = data.total_detections;
        
        // Crear gráfico de distribución de patrones
        createPatternsChart(data.pattern_distribution);
    }
    
    // Función para crear el gráfico de distribución de patrones
    function createPatternsChart(patternDistribution) {
        // Mapear tipos de patrones a nombres más amigables
        const patternNames = {
            'confirmshaming': 'Confirmshaming',
            'preselection': 'Preselección',
            'hidden_costs': 'Cargos ocultos',
            'difficult_cancellation': 'Difícil cancelación',
            'misleading_ads': 'Publicidad engañosa',
            'false_urgency': 'Falsa urgencia',
            'confusing_interface': 'Interfaces confusas'
        };
        
        // Preparar datos para el gráfico
        const labels = [];
        const data = [];
        const backgroundColors = [
            '#3498db', // Azul
            '#2ecc71', // Verde
            '#e74c3c', // Rojo
            '#f39c12', // Naranja
            '#9b59b6', // Morado
            '#1abc9c', // Turquesa
            '#34495e'  // Gris oscuro
        ];
        
        let i = 0;
        for (const patternType in patternDistribution) {
            labels.push(patternNames[patternType] || patternType);
            data.push(patternDistribution[patternType]);
            i++;
        }
        
        // Si no hay datos, mostrar mensaje
        if (data.every(value => value === 0)) {
            if (patternsChart) {
                const ctx = patternsChart.getContext('2d');
                ctx.font = '16px Arial';
                ctx.fillStyle = '#6c757d';
                ctx.textAlign = 'center';
                ctx.fillText('No hay datos disponibles', patternsChart.width / 2, patternsChart.height / 2);
            }
            return;
        }
        
        // Crear gráfico
        if (patternsChart) {
            new Chart(patternsChart, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Número de detecciones',
                        data: data,
                        backgroundColor: backgroundColors,
                        borderColor: backgroundColors.map(color => color),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                precision: 0
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const value = context.raw;
                                    const total = data.reduce((a, b) => a + b, 0);
                                    const percentage = total > 0 ? Math.round((value / total) * 100) : 0;
                                    return `${value} detecciones (${percentage}%)`;
                                }
                            }
                        }
                    }
                }
            });
        }
    }
});
