# Detailed Analysis Context

## Analysis Request
**Query**: {{ query }}
**Analysis Type**: {{ analysis_type | default('General') }}
**Depth Level**: {{ depth_level | default('Standard') }}

## Source Materials
{% if retrieved_nodes %}
{% for node in retrieved_nodes %}
### Source {{ loop.index }}: {{ node.metadata.get('title', 'Untitled') if node.metadata else 'Untitled' }}
**Relevance Score**: {{ "%.3f"|format(node.score) if node.score else 'N/A' }}
**Document Type**: {{ node.metadata.get('document_type', 'Unknown') if node.metadata else 'Unknown' }}
**Created**: {{ node.metadata.get('created_date', 'Unknown') if node.metadata else 'Unknown' }}

#### Content:
{{ node.text }}

#### Key Metadata:
{% if node.metadata %}
{% for key, value in node.metadata.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

---
{% endfor %}
{% else %}
No source materials available for analysis.
{% endif %}

## Analysis Parameters
- **Focus Areas**: {{ focus_areas | join(', ') if focus_areas else 'General analysis' }}
- **Required Depth**: {{ required_depth | default('Comprehensive') }}
- **Include Citations**: {{ include_citations | default(true) }}

## Response Guidelines
Provide a detailed analysis incorporating all relevant source materials. Include proper citations and cross-references where applicable.