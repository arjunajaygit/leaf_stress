"""User-facing summaries of experimental visual categories, not measured water status."""
SUMMARIES = {
    'Healthy': ('Low visible stress', 'The selected foliage resembles the model’s healthy examples. Early water stress may still be invisible.', 'Check soil moisture and continue observing the plant.'),
    'Moderate_Water_Stress': ('Moderate visual stress', 'The selected foliage resembles the moderate water-stress examples. Other growing conditions can produce similar symptoms.', 'Check soil moisture and inspect affected leaves before deciding whether to water.'),
    'Severe_Water_Stress': ('Severe visual stress', 'The selected foliage resembles the severe water-stress examples. The image alone cannot establish the cause.', 'Inspect the plant promptly and check soil moisture. Seek agricultural advice if symptoms persist.'),
    'Disease_Confounded': ('Other symptoms — water stress unclear', 'The image resembles the model’s disease or confounded examples. A water-stress level cannot be assigned from this result.', 'Inspect for pests, disease and nutrient issues; seek independent assessment.'),
}


def describe_result(prediction, top_probability):
    title, explanation, next_check = SUMMARIES[prediction]
    tentative = top_probability < .60
    return dict(title=title, explanation=explanation, next_check=next_check,
                tentative=tentative,
                reliability='Tentative visual estimate' if tentative else 'Experimental visual estimate',
                note='Confirm with plant inspection and a soil-moisture check. This prototype does not measure water status.')
