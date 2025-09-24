# Response Context

## Query Information
- **Query**: {{ query }}
- **Timestamp**: {{ timestamp }}
- **Response Mode**: {{ response_mode }}

## Retrieved Context
{% if retrieved_nodes %}
{% for node in retrieved_nodes %}
### Context {{ loop.index }}
**Score**: {{ node.score if node.score else 'N/A' }}
**Source**: {{ node.metadata.get('source', 'Unknown') if node.metadata else 'Unknown' }}

{{ node.text }}

---
{% endfor %}
{% else %}
No relevant context found.
{% endif %}

## Additional Information
{% if metadata %}
{% for key, value in metadata.items() %}
- **{{ key }}**: {{ value }}
{% endfor %}
{% endif %}

## Instructions
Based on the above context, provide a comprehensive and accurate response to the query. If the context doesn't contain sufficient information, clearly state what information is missing or unavailable.