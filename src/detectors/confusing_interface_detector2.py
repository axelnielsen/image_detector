        # Iniciar búsqueda desde la raíz con manejo de errores
        try:
            search_buttons_and_links(dom_structure)
        except Exception as e:
            print(f"Error al buscar botones y enlaces: {e}")
            return []
        
        # Analizar botones y enlaces para detectar inconsistencias
        for button in buttons_and_links:
            inconsistencies = []
            
            # Verificar inconsistencia entre texto y clase
            if button["is_primary_text"] and button["has_secondary_class"]:
                inconsistencies.append("Texto de acción primaria con clase de botón secundario")
            
            if button["is_secondary_text"] and button["has_primary_class"]:
                inconsistencies.append("Texto de acción secundaria con clase de botón primario")
            
            # Verificar si el botón de cancelar o rechazar tiene estilo visual prominente
            if button["is_secondary_text"] and button["has_primary_class"]:
                inconsistencies.append("Botón de cancelar/rechazar con estilo visual prominente")
            
            # Verificar si el botón de aceptar o confirmar tiene estilo visual poco prominente
            if button["is_primary_text"] and button["has_secondary_class"]:
                inconsistencies.append("Botón de aceptar/confirmar con estilo visual poco prominente")
            
            if inconsistencies:
                confidence = self.calculate_confidence(len(inconsistencies), 0.85)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "confusing_interface",
                        "evidence_type": "misleading_button",
                        "evidence": {
                            "path": button["path"],
                            "text": button["text"],
                            "classes": button["classes"],
                            "inconsistencies": inconsistencies
                        },
                        "confidence": confidence,
                        "location": f"Botón engañoso en {button['path']}",
                        "screenshot": screenshot_path
                    })
        
        # 2. Buscar grupos de botones con jerarquía visual confusa
        # Agrupar botones que están cerca en el DOM
        button_groups = {}
        
        for i, button in enumerate(buttons_and_links):
            # Extraer el path del padre
            parent_path = " > ".join(button["path"].split(" > ")[:-1])
            
            if parent_path not in button_groups:
                button_groups[parent_path] = []
            
            button_groups[parent_path].append(button)
        
        # Analizar cada grupo de botones
        for parent_path, group in button_groups.items():
            if len(group) >= 2:  # Al menos dos botones en el grupo
                primary_buttons = [b for b in group if b["is_primary_text"] or b["has_primary_class"]]
                secondary_buttons = [b for b in group if b["is_secondary_text"] or b["has_secondary_class"]]
                
                # Verificar si hay múltiples botones primarios
                if len(primary_buttons) > 1:
                    confidence = self.calculate_confidence(len(primary_buttons), 0.75)
                    
                    if confidence >= self.confidence_threshold:
                        detections.append({
                            "pattern_type": "confusing_interface",
                            "evidence_type": "multiple_primary_buttons",
                            "evidence": {
                                "parent_path": parent_path,
                                "buttons": [
                                    {
                                        "path": b["path"],
                                        "text": b["text"],
                                        "classes": b["classes"]
                                    } 
                                    for b in primary_buttons
                                ]
                            },
                            "confidence": confidence,
                            "location": f"Múltiples botones primarios en {parent_path}",
                            "screenshot": screenshot_path
                        })
                
                # Verificar si hay botones primarios y secundarios con estilos similares
                if primary_buttons and secondary_buttons:
                    # Comparar clases para ver si son visualmente similares
                    similar_styles = False
                    
                    # Simplificación: si no hay clases distintivas, asumimos que son visualmente similares
                    primary_distinctive = any(b["has_primary_class"] for b in primary_buttons)
                    secondary_distinctive = any(b["has_secondary_class"] for b in secondary_buttons)
                    
                    if not (primary_distinctive and secondary_distinctive):
                        similar_styles = True
                    
                    if similar_styles:
                        confidence = self.calculate_confidence(len(primary_buttons) + len(secondary_buttons), 0.8)
                        
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                "pattern_type": "confusing_interface",
                                "evidence_type": "similar_button_styles",
                                "evidence": {
                                    "parent_path": parent_path,
                                    "primary_buttons": [
                                        {
                                            "path": b["path"],
                                            "text": b["text"],
                                            "classes": b["classes"]
                                        } 
                                        for b in primary_buttons
                                    ],
                                    "secondary_buttons": [
                                        {
                                            "path": b["path"],
                                            "text": b["text"],
                                            "classes": b["classes"]
                                        } 
                                        for b in secondary_buttons
                                    ]
                                },
                                "confidence": confidence,
                                "location": f"Botones con estilos similares en {parent_path}",
                                "screenshot": screenshot_path
                            })
