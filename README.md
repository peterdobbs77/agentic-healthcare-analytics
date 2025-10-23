# agentic-healthcare-analytics
Agentic AI system for performing analytics on healthcare data

## Prerequisites

* 🔥 [Folder of FHIR Resources](./fhir/) -- this is excluded from the repo via `.gitignore` but can be populated by your own FHIR Resources or with a synthetic data generator like [synthea](https://synthea.mitre.org/).
* Python 3.12+
* Various python libraries, for which I attempt to include installation steps in each playbook. Apologies if I missed any. I'll provide a virtual environment at some point.

## Folder organization

Start with the playbooks in [FHIR_RAG](./FHIR_RAG/) to get a basis for the investigations. I build from there towards using Knowledge Graphs under [FHIR_KNOWLEDGE_GRAPHS](./FHIR_KNOWLEDGE_GRAPHS/).