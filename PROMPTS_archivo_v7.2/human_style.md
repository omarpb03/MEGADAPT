### Este bloque va al final de cada prompt para obligar al LLM a proporcionar las etiquetas cortas y el JSON

"FORMATO:

Cada elemento debe ser una frase corta de 4 o 5 palabras (máximo 6), en forma de etiqueta, no una oración completa.

Ejemplos del estilo esperado: "Escasez de agua", "Tandeo en las colonias", "Fugas en la red", "Asentamientos irregulares", "Daños a la salud", "Compra de agua en pipas".

Ejemplos de acciones con actor: {"actor": "Protección Civil", "accion": "Emite recomendaciones de riesgo"}, {"actor": "Vecinos", "accion": "Almacenan agua en cisternas"}.

No repitas la misma idea dos veces dentro de un mismo campo.

No agregues información que no esté en la transcripción. Si un campo no se menciona, devuelve lista vacía.

Todo en español. Devuelve únicamente el JSON con la estructura solicitada."