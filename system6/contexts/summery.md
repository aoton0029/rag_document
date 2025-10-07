# Summary Response Context

## Query: {{ query }}

## Key Information Summary
{% if retrieved_nodes %}
{% for node in retrieved_nodes %}
**Summary {{ loop.index }}** (Score: {{ "%.2f"|format(node.score) if node.score else 'N/A' }}):
{{ node.text[:200] }}...

{% endfor %}
{% else %}
No relevant information found for summarization.
{% endif %}

## Response Instructions
Provide a concise summary based on the retrieved information. Focus on the most important points and maintain accuracy.

{% if user_preferences %}
**User Preferences**: {{ user_preferences }}
{% endif %}