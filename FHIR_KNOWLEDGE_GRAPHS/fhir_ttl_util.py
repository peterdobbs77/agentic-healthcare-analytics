from rdflib import Dataset

def get_conditions_foreach_patient(graph: Dataset) -> list[tuple[str, str, str]]:
    """Retrieves list of conditions for each patient"""
    query="""
        PREFIX fhir: <http://hl7.org/fhir/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?patientId ?conditionCode ?conditionDisplay
        WHERE {
            ?condition rdf:type fhir:Condition .

            # Link to the patient
            ?condition fhir:subject ?subjectRef .
            # ?subjectRef fhir:reference ?patientRef .
            BIND(STRAFTER(STR(?subjectRef), "Patient/") AS ?patientId)

            # Get the condition code
            ?condition fhir:code ?codeNode .
            ?codeNode fhir:coding ?codingNode .
            ?codingNode fhir:code ?conditionCode .
            OPTIONAL { ?codingNode fhir:display ?conditionDisplay }
        }
        """
    
    return [
        (str(row.patientId), str(row.conditionCode), str(row.conditionDisplay))
        for row in graph.query(query)
    ]