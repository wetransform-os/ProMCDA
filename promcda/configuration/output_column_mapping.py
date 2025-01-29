from promcda.enums import NormalizationFunctions, AggregationFunctions, OutputColumnNames4Sensitivity

output_column_mapping = {
    (AggregationFunctions.WEIGHTED_SUM.value, NormalizationFunctions.MINMAX.value): OutputColumnNames4Sensitivity.WS_MINMAX_01.value,
    (AggregationFunctions.WEIGHTED_SUM.value, NormalizationFunctions.TARGET.value): OutputColumnNames4Sensitivity.WS_TARGET_01.value,
    (AggregationFunctions.WEIGHTED_SUM.value, NormalizationFunctions.STANDARDIZED.value): OutputColumnNames4Sensitivity.WS_STANDARDIZED_ANY.value,
    (AggregationFunctions.WEIGHTED_SUM.value, NormalizationFunctions.RANK.value): OutputColumnNames4Sensitivity.WS_RANK.value,
    (AggregationFunctions.GEOMETRIC.value, NormalizationFunctions.MINMAX.value): OutputColumnNames4Sensitivity.GEOM_MINMAX_WITHOUT_ZERO.value,
    (AggregationFunctions.GEOMETRIC.value, NormalizationFunctions.TARGET.value): OutputColumnNames4Sensitivity.GEOM_TARGET_WITHOUT_ZERO.value,
    (AggregationFunctions.GEOMETRIC.value, NormalizationFunctions.STANDARDIZED.value): OutputColumnNames4Sensitivity.GEOM_STANDARDIZED_WITHOUT_ZERO.value,
    (AggregationFunctions.GEOMETRIC.value, NormalizationFunctions.RANK.value): OutputColumnNames4Sensitivity.GEOM_RANK.value,
    (AggregationFunctions.HARMONIC.value, NormalizationFunctions.MINMAX.value): OutputColumnNames4Sensitivity.HARM_MINMAX_WITHOUT_ZERO.value,
    (AggregationFunctions.HARMONIC.value, NormalizationFunctions.TARGET.value): OutputColumnNames4Sensitivity.HARM_TARGET_WITHOUT_ZERO.value,
    (AggregationFunctions.HARMONIC.value, NormalizationFunctions.STANDARDIZED.value): OutputColumnNames4Sensitivity.HARM_STANDARDIZED_WITHOUT_ZERO.value,
    (AggregationFunctions.HARMONIC.value, NormalizationFunctions.RANK.value): OutputColumnNames4Sensitivity.HARM_RANK.value,
    (AggregationFunctions.MINIMUM.value, NormalizationFunctions.STANDARDIZED.value): OutputColumnNames4Sensitivity.MIN_STANDARDIZED_ANY.value,
}