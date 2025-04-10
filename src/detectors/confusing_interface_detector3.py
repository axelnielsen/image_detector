        # 3. Buscar elementos de interfaz que puedan ser confusos
        # Buscar elementos que parezcan controles de interfaz
        ui_elements = []
        
        def search_ui_elements(node, path="body"):
            # Verificar que el nodo es un diccionario válido
            if not isinstance(node, dict):
                return
                
            # Verificar si el nodo es un elemento de interfaz
            node_type = node.get("type")
            if not node_type:
                return
                
            is_ui_element = node_type in ["INPUT", "SELECT", "TEXTAREA", "LABEL", "FORM"]
            
            if is_ui_element:
                ui_elements.append({
                    "node": node,
                    "path": path,
                    "type": node_type,
                    "attributes": node.get("attributes", {}),
                    "classes": node.get("classes", []),
                    "text": node.get("text", "")
                })
            
            # Buscar en nodos hijos
            children = node.get("children", [])
            if children and isinstance(children, list):
                for i, child in enumerate(children):
                    if isinstance(child, dict):
                        child_type = child.get("type", "unknown")
                        child_path = f"{path} > {child_type}[{i}]"
                        search_ui_elements(child, child_path)
        
        # Iniciar búsqueda desde la raíz con manejo de errores
        try:
            search_ui_elements(dom_structure)
        except Exception as e:
            print(f"Error al buscar elementos de interfaz: {e}")
            return detections  # Devolver las detecciones que ya tenemos
        
        # Analizar elementos de interfaz para detectar confusiones
        for element in ui_elements:
            confusing_aspects = []
            
            # Verificar si es un checkbox o radio sin label claro
            if element["type"] == "INPUT" and element["attributes"].get("type") in ["checkbox", "radio"]:
                # Buscar label asociado
                has_label = False
                
                # Verificar si tiene ID
                if "id" in element["attributes"]:
                    element_id = element["attributes"]["id"]
                    
                    # Buscar label con atributo "for" que coincida con el ID
                    for ui_el in ui_elements:
                        if ui_el["type"] == "LABEL" and ui_el["attributes"].get("for") == element_id:
                            has_label = True
                            break
                
                if not has_label:
                    confusing_aspects.append("Checkbox o radio sin label claro")
            
            # Verificar si es un input con placeholder pero sin label
            if element["type"] == "INPUT" and element["attributes"].get("placeholder") and element["attributes"].get("type") not in ["hidden", "submit", "button"]:
                # Buscar label asociado
                has_label = False
                
                # Verificar si tiene ID
                if "id" in element["attributes"]:
                    element_id = element["attributes"]["id"]
                    
                    # Buscar label con atributo "for" que coincida con el ID
                    for ui_el in ui_elements:
                        if ui_el["type"] == "LABEL" and ui_el["attributes"].get("for") == element_id:
                            has_label = True
                            break
                
                if not has_label:
                    confusing_aspects.append("Input con placeholder pero sin label")
            
            # Verificar si es un select sin label claro
            if element["type"] == "SELECT":
                # Buscar label asociado
                has_label = False
                
                # Verificar si tiene ID
                if "id" in element["attributes"]:
                    element_id = element["attributes"]["id"]
                    
                    # Buscar label con atributo "for" que coincida con el ID
                    for ui_el in ui_elements:
                        if ui_el["type"] == "LABEL" and ui_el["attributes"].get("for") == element_id:
                            has_label = True
                            break
                
                if not has_label:
                    confusing_aspects.append("Select sin label claro")
            
            if confusing_aspects:
                confidence = self.calculate_confidence(len(confusing_aspects), 0.7)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "confusing_interface",
                        "evidence_type": "confusing_ui_element",
                        "evidence": {
                            "path": element["path"],
                            "type": element["type"],
                            "attributes": element["attributes"],
                            "confusing_aspects": confusing_aspects
                        },
                        "confidence": confidence,
                        "location": f"Elemento de interfaz confuso en {element['path']}",
                        "screenshot": screenshot_path
                    })
        
        # 4. Buscar formularios con diseño confuso
        forms = []
        
        def search_forms(node, path="body"):
            # Verificar que el nodo es un diccionario válido
            if not isinstance(node, dict):
                return
                
            # Verificar si el nodo es un formulario
            node_type = node.get("type")
            if not node_type:
                return
                
            if node_type == "FORM":
                forms.append({
                    "node": node,
                    "path": path,
                    "attributes": node.get("attributes", {}),
                    "classes": node.get("classes", [])
                })
            
            # Buscar en nodos hijos
            children = node.get("children", [])
            if children and isinstance(children, list):
                for i, child in enumerate(children):
                    if isinstance(child, dict):
                        child_type = child.get("type", "unknown")
                        child_path = f"{path} > {child_type}[{i}]"
                        search_forms(child, child_path)
        
        # Iniciar búsqueda desde la raíz con manejo de errores
        try:
            search_forms(dom_structure)
        except Exception as e:
            print(f"Error al buscar formularios: {e}")
            return detections  # Devolver las detecciones que ya tenemos
        
        # Analizar formularios para detectar confusiones
        for form in forms:
            confusing_aspects = []
            
            # Verificar si el formulario tiene un botón de envío claro
            has_submit_button = False
            
            for button in buttons_and_links:
                if button["path"].startswith(form["path"]):
                    if (button["button_type"] == "submit" or 
                        button["is_primary_text"] or 
                        button["has_primary_class"]):
                        has_submit_button = True
                        break
            
            if not has_submit_button:
                confusing_aspects.append("Formulario sin botón de envío claro")
            
            if confusing_aspects:
                confidence = self.calculate_confidence(len(confusing_aspects), 0.75)
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "pattern_type": "confusing_interface",
                        "evidence_type": "confusing_form",
                        "evidence": {
                            "path": form["path"],
                            "attributes": form["attributes"],
                            "confusing_aspects": confusing_aspects
                        },
                        "confidence": confidence,
                        "location": f"Formulario confuso en {form['path']}",
                        "screenshot": screenshot_path
                    })
        
        return detections
    
    def calculate_confidence(self, evidence_count: int, base_confidence: float) -> float:
        """
        Calcula la confianza basada en la cantidad de evidencia.
        
        Args:
            evidence_count: Cantidad de evidencias encontradas
            base_confidence: Confianza base para este tipo de patrón
            
        Returns:
            float: Valor de confianza entre 0 y 1
        """
        # Ajustar confianza según la cantidad de evidencia
        confidence = base_confidence
        
        if evidence_count > 1:
            # Aumentar confianza si hay múltiples evidencias
            confidence += min(0.2, evidence_count * 0.05)
        
        # Asegurar que la confianza no supere 1.0
        return min(confidence, 1.0)
