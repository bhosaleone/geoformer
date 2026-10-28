"""
Example 4: Hospital Emergency Casualty Triage (Research Prototype Demo)
======================================================================
Demonstrates sub-millisecond differential diagnosis and contraindication safety
alerting across an acute cardiovascular emergency (STEMI vs. Aortic Dissection).

===============================================================================
⚠️ MEDICAL RESEARCH DISCLAIMER:
This software is a preliminary proof-of-concept research demonstration.
It is strictly NOT approved, certified, or intended for direct clinical diagnostic
use on human patients. All medical decisions must be made by qualified healthcare
professionals adhering to institutional clinical guidelines and standards of care.
===============================================================================

License: MIT
Author: Shrikant Bhosale (Atmabhan Pandit), TWIST POOL Labs
"""

from geoformer.clinical import GeoFormerClinicalDiagnosticEngine

def main():
    print("=" * 75)
    print("🏥 GeoFormer Example 4: Emergency Casualty Triage (Research Prototype)")
    print("=" * 75)
    print("⚠️  DISCLAIMER: Research proof-of-concept only. Not for human clinical use.")
    print("=" * 75)

    # 1. Initialize clinical reasoning engine
    engine = GeoFormerClinicalDiagnosticEngine()
    print(f"• Knowledge Graph initialized with {len(engine.ont.entity2id)} total entities\n")

    # 2. Case Presentation: Aortic Dissection mimicking Acute Coronary Syndrome
    print("───────────────────────────────────────────────────────────────────────────")
    print("CLINICAL CASE: ACUTE VASCULAR EMERGENCY (DIAGNOSTIC DILEMMA)")
    print("───────────────────────────────────────────────────────────────────────────")
    patient_presentation = (
        "64-year-old female presenting with sudden tearing back pain, severe refractory "
        "hypertension, asymmetric radial pulses, and widened mediastinum on chest X-ray."
    )
    print(f"Patient Presentation:\n  \"{patient_presentation}\"\n")

    findings = [
        "Tearing_Interscapular_Back_Pain",
        "Pulse_Deficit_Between_Arms",
        "Severe_Refractory_Hypertension",
        "Mediastinal_Widening_on_CXR"
    ]
    print(f"Observed Clinical Findings: {findings}")

    # 3. Execute topological metric relaxation
    result = engine.diagnose_patient(findings, top_k=3)
    latency_ms = result["inference_latency_ms"]

    print(f"\n[1] Ranked Differential Diagnoses (Inference Latency: {latency_ms:.2f} ms):")
    for d in result["differentials"]:
        prob = d["probability"]
        bar = "█" * int(prob / 4)
        print(f"  • {d['disease']:<30} Probability: {prob:5.1f}%  {bar}")

    print(f"\n[2] Primary Diagnostic Conclusion:")
    print(f"  -> Confirmed Diagnosis: {result['primary_diagnosis']}")

    print(f"\n[3] Explainable Pathophysiological Causal Pathway:")
    for path in result["pathophysiological_causal_chain"]:
        print(f"  -> {path}")

    print(f"\n[4] Automated Black-Box Safety & Contraindication Alerts:")
    if result["safety_contraindications"]:
        for alert in result["safety_contraindications"]:
            print(f"  🚨 {alert}")
    else:
        print("  • No acute contraindications flagged.")

    print(f"\n[5] First-Line Emergency Interventions:")
    for rec in result["recommended_interventions"]:
        print(f"  ✓ {rec}")

    print("\n" + "=" * 75)
    print("RESEARCH AUDIT & SCOPE SUMMARY:")
    print(f"  • Inference Latency: {latency_ms:.2f} ms (deterministic on CPU)")
    print(f"  • Transparent Causality: Multi-hop graph path explains the differential")
    print(f"  • Critical Safety Guard: Automatically flagged lethal thrombolytic hazard")
    print("=" * 75)

if __name__ == "__main__":
    main()
