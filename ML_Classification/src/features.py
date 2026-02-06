"""
Central registry for SPAPS features & confounds.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd


# ---- Target & confounds ----
CONFOUNDS: Tuple[str, ...] = ("Alter", "Geschlecht", "TIV_NEU")


# ---- Feature sets ----
FEATURE_SETS: Dict[str, List[str]] = {
    # Speech — 88 Acoustic features (prosody, spectral, MFCCs, formants, jitter/shimmer, etc.)
    "ACOUSTIC_FEATURES": [
        # F0/prosody (10 features)
        "F0semitoneFrom27.5Hz_sma3nz_amean",
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
        "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
        "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
        "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
        "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
        "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
        # Loudness (10 features)
        "loudness_sma3_amean",
        "loudness_sma3_stddevNorm",
        "loudness_sma3_percentile20.0",
        "loudness_sma3_percentile50.0",
        "loudness_sma3_percentile80.0",
        "loudness_sma3_pctlrange0-2",
        "loudness_sma3_meanRisingSlope",
        "loudness_sma3_stddevRisingSlope",
        "loudness_sma3_meanFallingSlope",
        "loudness_sma3_stddevFallingSlope",
        # Spectral + MFCCs (10 features)
        "spectralFlux_sma3_amean",
        "spectralFlux_sma3_stddevNorm",
        "mfcc1_sma3_amean",
        "mfcc1_sma3_stddevNorm",
        "mfcc2_sma3_amean",
        "mfcc2_sma3_stddevNorm",
        "mfcc3_sma3_amean",
        "mfcc3_sma3_stddevNorm",
        "mfcc4_sma3_amean",
        "mfcc4_sma3_stddevNorm",
        # Voice quality (6 features)
        "jitterLocal_sma3nz_amean",
        "jitterLocal_sma3nz_stddevNorm",
        "shimmerLocaldB_sma3nz_amean",
        "shimmerLocaldB_sma3nz_stddevNorm",
        "HNRdBACF_sma3nz_amean",
        "HNRdBACF_sma3nz_stddevNorm",
        # Harmonic energy ratios (4 features)
        "logRelF0-H1-H2_sma3nz_amean",
        "logRelF0-H1-H2_sma3nz_stddevNorm",
        "logRelF0-H1-A3_sma3nz_amean",
        "logRelF0-H1-A3_sma3nz_stddevNorm",
        # Formants (18 features)
        "F1frequency_sma3nz_amean",
        "F1frequency_sma3nz_stddevNorm",
        "F1bandwidth_sma3nz_amean",
        "F1bandwidth_sma3nz_stddevNorm",
        "F1amplitudeLogRelF0_sma3nz_amean",
        "F1amplitudeLogRelF0_sma3nz_stddevNorm",
        "F2frequency_sma3nz_amean",
        "F2frequency_sma3nz_stddevNorm",
        "F2bandwidth_sma3nz_amean",
        "F2bandwidth_sma3nz_stddevNorm",
        "F2amplitudeLogRelF0_sma3nz_amean",
        "F2amplitudeLogRelF0_sma3nz_stddevNorm",
        "F3frequency_sma3nz_amean",
        "F3frequency_sma3nz_stddevNorm",
        "F3bandwidth_sma3nz_amean",
        "F3bandwidth_sma3nz_stddevNorm",
        "F3amplitudeLogRelF0_sma3nz_amean",
        "F3amplitudeLogRelF0_sma3nz_stddevNorm",
        # Voiced / unvoiced spectral variants (23 features)
        "alphaRatioV_sma3nz_amean",
        "alphaRatioV_sma3nz_stddevNorm",
        "hammarbergIndexV_sma3nz_amean",
        "hammarbergIndexV_sma3nz_stddevNorm",
        "slopeV0-500_sma3nz_amean",
        "slopeV0-500_sma3nz_stddevNorm",
        "slopeV500-1500_sma3nz_amean",
        "slopeV500-1500_sma3nz_stddevNorm",
        "spectralFluxV_sma3nz_amean",
        "spectralFluxV_sma3nz_stddevNorm",
        "mfcc1V_sma3nz_amean",
        "mfcc1V_sma3nz_stddevNorm",
        "mfcc2V_sma3nz_amean",
        "mfcc2V_sma3nz_stddevNorm",
        "mfcc3V_sma3nz_amean",
        "mfcc3V_sma3nz_stddevNorm",
        "mfcc4V_sma3nz_amean",
        "mfcc4V_sma3nz_stddevNorm",
        "alphaRatioUV_sma3nz_amean",
        "hammarbergIndexUV_sma3nz_amean",
        "slopeUV0-500_sma3nz_amean",
        "slopeUV500-1500_sma3nz_amean",
        "spectralFluxUV_sma3nz_amean",
        # Segment-level stats (7 features)
        "loudnessPeaksPerSec",
        "VoicedSegmentsPerSec",
        "MeanVoicedSegmentLengthSec",
        "StddevVoicedSegmentLengthSec",
        "MeanUnvoicedSegmentLength",
        "StddevUnvoicedSegmentLength",
        "equivalentSoundLevel_dBp"
    ],
    # Speech — 51 Transcript features (lexical, POS, coherence, graph metrics, etc.)
    "TRANSCRIPT_FEATURES": [
        # Counts & length (why: basic scale/verbosity)
        "Total_Words", "Total_Sentences", "MLU",
        # Lexical diversity (why: vocabulary richness)
        "TTR", "MTLD",
        # POS & ratio features (why: morpho-syntactic usage pattern)
        "ADJ_Ratio", "ADV_Ratio", "DET_Ratio", "NOUN_Ratio", "VERB_Ratio",
        "PRON_Ratio", "Pers_PRON_Ratio", "NVR",
        # Syntactic complexity (why: structural sophistication)
        "SynC", "SynD", "Syn_complexity", "Mean_Dependency_Distance", "Subordination_Ratio",
        # Cohesion / coherence – surface (why: explicit linking and overlap)
        "Connective_Ratio", "OCR", "Root_overlap", "Word_Level_Coh", "Sentence_Level_Coh",
        "Graph_based_Cohesion", "SimS_R",
        # Embedding-based coherence (why: semantic relatedness beyond surface)
        "FastText_Similarity_mean", "BERT_Coherence_mean", "BioLord_Coherence_mean", "Hohenheimer_Index",
        # Semantics / density & readability (why: information packing & accessibility)
        "Semantic_Coherence", "Semantic_Density", "Readability",
        # Disfluencies & errors (why: fluency/quality signals)
        "Filled_Pauses", "Repetitions", "Grammatical_Error",
        # Sentiment (why: affective tone)
        "Negative_Sentiment", "Probability_Negative",
        # Speech-graph metrics (why: discourse network structure)
        "N", "E", "PE", "LSCC", "ATD", "L1", "L2", "L3"
    ],
    # Neurocognition (27)
    "NEUROCOGNITIVE_FEATURES": [
        "Tiere_Sum",
        "Tiere_Pers",
        "Tiere_Fehler",
        "P_Sum",
        "P_Pers",
        "P_Fehler",
        "Altern_Sum",
        "Altern_Pers",
        "Altern_Fehler",
        "VLMT_Sum_Richtige",
        "VLMT_VerlustInteferenz",
        "VLMT_VerlustZeitlicheVerzoegerung",
        "VLMT_RecognitionCorrected",
        "VLMT_Sum_FalschPositive",
        "VLMT_Sum_Perseverationen",
        "VLMT_Sum_Interferenz",
        "BZT_Sum",
        "Blockspanne_Vorwaerts", "Blockspanne_Rueckwaerts",
        "PfadfinderA", "PfadfinderB",
        "TMT_Differenz",
        "D2_GZ", "D2_F1", "D2_F2", "D2_KL",
        "MWTB_Sum"
    ],

    # sMRI
    "SMRI_FEATURES": [
        # SURFACE
        "L_bankssts_surfavg",
        "L_caudalanteriorcingulate_surfavg",
        "L_caudalmiddlefrontal_surfavg",
        "L_cuneus_surfavg",
        "L_entorhinal_surfavg",
        "L_fusiform_surfavg",
        "L_inferiorparietal_surfavg",
        "L_inferiortemporal_surfavg",
        "L_isthmuscingulate_surfavg",
        "L_lateraloccipital_surfavg",
        "L_lateralorbitofrontal_surfavg",
        "L_lingual_surfavg",
        "L_medialorbitofrontal_surfavg",
        "L_middletemporal_surfavg",
        "L_parahippocampal_surfavg",
        "L_paracentral_surfavg",
        "L_parsopercularis_surfavg",
        "L_parsorbitalis_surfavg",
        "L_parstriangularis_surfavg",
        "L_pericalcarine_surfavg",
        "L_postcentral_surfavg",
        "L_posteriorcingulate_surfavg",
        "L_precentral_surfavg",
        "L_precuneus_surfavg",
        "L_rostralanteriorcingulate_surfavg",
        "L_rostralmiddlefrontal_surfavg",
        "L_superiorfrontal_surfavg",
        "L_superiorparietal_surfavg",
        "L_superiortemporal_surfavg",
        "L_supramarginal_surfavg",
        "L_frontalpole_surfavg",
        "L_temporalpole_surfavg",
        "L_transversetemporal_surfavg",
        "L_insula_surfavg",
        "R_bankssts_surfavg",
        "R_caudalanteriorcingulate_surfavg",
        "R_caudalmiddlefrontal_surfavg",
        "R_cuneus_surfavg",
        "R_entorhinal_surfavg",
        "R_fusiform_surfavg",
        "R_inferiorparietal_surfavg",
        "R_inferiortemporal_surfavg",
        "R_isthmuscingulate_surfavg",
        "R_lateraloccipital_surfavg",
        "R_lateralorbitofrontal_surfavg",
        "R_lingual_surfavg",
        "R_medialorbitofrontal_surfavg",
        "R_middletemporal_surfavg",
        "R_parahippocampal_surfavg",
        "R_paracentral_surfavg",
        "R_parsopercularis_surfavg",
        "R_parsorbitalis_surfavg",
        "R_parstriangularis_surfavg",
        "R_pericalcarine_surfavg",
        "R_postcentral_surfavg",
        "R_posteriorcingulate_surfavg",
        "R_precentral_surfavg",
        "R_precuneus_surfavg",
        "R_rostralanteriorcingulate_surfavg",
        "R_rostralmiddlefrontal_surfavg",
        "R_superiorfrontal_surfavg",
        "R_superiorparietal_surfavg",
        "R_superiortemporal_surfavg",
        "R_supramarginal_surfavg",
        "R_frontalpole_surfavg",
        "R_temporalpole_surfavg",
        "R_transversetemporal_surfavg",
        "R_insula_surfavg",
        # THICKNESS
        "L_bankssts_thickavg",
        "L_caudalanteriorcingulate_thickavg",
        "L_caudalmiddlefrontal_thickavg",
        "L_cuneus_thickavg",
        "L_entorhinal_thickavg",
        "L_fusiform_thickavg",
        "L_inferiorparietal_thickavg",
        "L_inferiortemporal_thickavg",
        "L_isthmuscingulate_thickavg",
        "L_lateraloccipital_thickavg",
        "L_lateralorbitofrontal_thickavg",
        "L_lingual_thickavg",
        "L_medialorbitofrontal_thickavg",
        "L_middletemporal_thickavg",
        "L_parahippocampal_thickavg",
        "L_paracentral_thickavg",
        "L_parsopercularis_thickavg",
        "L_parsorbitalis_thickavg",
        "L_parstriangularis_thickavg",
        "L_pericalcarine_thickavg",
        "L_postcentral_thickavg",
        "L_posteriorcingulate_thickavg",
        "L_precentral_thickavg",
        "L_precuneus_thickavg",
        "L_rostralanteriorcingulate_thickavg",
        "L_rostralmiddlefrontal_thickavg",
        "L_superiorfrontal_thickavg",
        "L_superiorparietal_thickavg",
        "L_superiortemporal_thickavg",
        "L_supramarginal_thickavg",
        "L_frontalpole_thickavg",
        "L_temporalpole_thickavg",
        "L_transversetemporal_thickavg",
        "L_insula_thickavg",
        "R_bankssts_thickavg",
        "R_caudalanteriorcingulate_thickavg",
        "R_caudalmiddlefrontal_thickavg",
        "R_cuneus_thickavg",
        "R_entorhinal_thickavg",
        "R_fusiform_thickavg",
        "R_inferiorparietal_thickavg",
        "R_inferiortemporal_thickavg",
        "R_isthmuscingulate_thickavg",
        "R_lateraloccipital_thickavg",
        "R_lateralorbitofrontal_thickavg",
        "R_lingual_thickavg",
        "R_medialorbitofrontal_thickavg",
        "R_middletemporal_thickavg",
        "R_parahippocampal_thickavg",
        "R_paracentral_thickavg",
        "R_parsopercularis_thickavg",
        "R_parsorbitalis_thickavg",
        "R_parstriangularis_thickavg",
        "R_pericalcarine_thickavg",
        "R_postcentral_thickavg",
        "R_posteriorcingulate_thickavg",
        "R_precentral_thickavg",
        "R_precuneus_thickavg",
        "R_rostralanteriorcingulate_thickavg",
        "R_rostralmiddlefrontal_thickavg",
        "R_superiorfrontal_thickavg",
        "R_superiorparietal_thickavg",
        "R_superiortemporal_thickavg",
        "R_supramarginal_thickavg",
        "R_frontalpole_thickavg",
        "R_temporalpole_thickavg",
        "R_transversetemporal_thickavg",
        "R_insula_thickavg",
        # SUBCORT
        "LLatVent", "RLatVent", "Lthal", "Rthal", "Lcaud", "Rcaud",
        "Lput", "Rput", "Lpal", "Rpal", "Lhippo", "Rhippo",
        "Lamyg", "Ramyg", "Laccumb", "Raccumb",
    ],

    "CONNECTIVITY_FEATURES": [
        "NumEdges",
        "TotalStreamlines",
        "MeanStreamlinesPerEdge",
        "Mean_RC",
        "Mean_Feeder",
        "Mean_Local",
        "C_real",
        "L_real",
        "GE_real",
        "C_norm",
        "L_norm",
        "GE_norm",
        "SW_index",
        "SW_sigma_classic"
    ]

}


## ---- Modality dataclass (shared with pipelines) ----
@dataclass(frozen=True)
class Modality:
    name: str
    columns: Tuple[str, ...]


def make_modalities(df: pd.DataFrame) -> Tuple[Modality, ...]:
    """
    Build modality objects from FEATURE_SETS, filtered to present columns.
    Keeps missing features out without failing the run.
    """
    def keep_existing(names: List[str]) -> Tuple[str, ...]:
        return tuple([c for c in names if c in df.columns])

    mods = [
        Modality("speech_acoustic", keep_existing(FEATURE_SETS.get("ACOUSTIC_FEATURES", []))),
        Modality("speech_text", keep_existing(FEATURE_SETS.get("TRANSCRIPT_FEATURES", []))),
        Modality("neurocog", keep_existing(FEATURE_SETS.get("NEUROCOGNITIVE_FEATURES", []))),
        Modality("smri", keep_existing(FEATURE_SETS.get("SMRI_FEATURES", []))),
        Modality("network", keep_existing(FEATURE_SETS.get("CONNECTIVITY_FEATURES", []))),
    ]
    # Drop empty modalities silently to allow partial-modality runs (e.g., no sMRI rows)
    return tuple([m for m in mods if len(m.columns) > 0])



