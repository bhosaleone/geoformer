"""
Clinical & Biomedical Ontology Engine
=====================================
Structured clinical knowledge graph encoding real medical diseases, symptoms,
biomarkers, pathophysiological mechanisms, drugs, and contraindications.

Derived from Harrison's Internal Medicine & Robbins Pathologic Basis of Disease.
Author: Shrikant Bhosale (Atmabhan Pandit), MBBS GMC Sambhajinagar
"""

from typing import Dict, List, Tuple, Set
import torch


class MedicalOntology:
    """Manages medical entity vocabularies and multi-relational edges."""
    def __init__(self):
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.entity_type: Dict[str, str] = {}
        
        self.relation2id = {
            "presents_with": 0,       # Disease -> Symptom
            "caused_by": 1,           # Symptom/Disease -> Pathophysiological mechanism
            "indicated_biomarker": 2, # Disease -> Lab/ECG/Imaging finding
            "treated_by": 3,          # Disease -> Drug/Intervention
            "contraindicated_in": 4,  # Drug -> Condition
            "complication_of": 5      # Disease A -> Disease B
        }
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        
        # Edges grouped by relation: rel_id -> list of (src_id, tgt_id)
        self.edges_by_rel: Dict[int, List[Tuple[int, int]]] = {r: [] for r in self.relation2id.values()}

    def add_entity(self, name: str, etype: str) -> int:
        if name not in self.entity2id:
            eid = len(self.entity2id)
            self.entity2id[name] = eid
            self.id2entity[eid] = name
            self.entity_type[name] = etype
        return self.entity2id[name]

    def add_triple(self, head: str, rel: str, tail: str, head_type: str = "concept", tail_type: str = "concept"):
        hid = self.add_entity(head, head_type)
        tid = self.add_entity(tail, tail_type)
        rid = self.relation2id[rel]
        self.edges_by_rel[rid].append((hid, tid))

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)


def build_clinical_knowledge_graph() -> MedicalOntology:
    """Constructs real clinical knowledge graph across acute and chronic medicine."""
    ont = MedicalOntology()

    # =========================================================================
    # 1. CARDIOLOGY
    # =========================================================================
    # STEMI (Inferior & Anterior)
    ont.add_triple("Acute_STEMI", "presents_with", "Retrosternal_Crushing_Chest_Pain", "disease", "symptom")
    ont.add_triple("Acute_STEMI", "presents_with", "Diaphoresis", "disease", "symptom")
    ont.add_triple("Acute_STEMI", "presents_with", "Radiation_to_Left_Arm", "disease", "symptom")
    ont.add_triple("Acute_STEMI", "presents_with", "Dyspnea", "disease", "symptom")
    ont.add_triple("Acute_STEMI", "indicated_biomarker", "Elevated_Cardiac_Troponin_I", "disease", "biomarker")
    ont.add_triple("Acute_STEMI", "indicated_biomarker", "ST_Segment_Elevation", "disease", "biomarker")
    ont.add_triple("Acute_STEMI", "caused_by", "Coronary_Atherosclerotic_Plaque_Rupture", "disease", "pathophysiology")
    ont.add_triple("Coronary_Atherosclerotic_Plaque_Rupture", "caused_by", "Acute_Occlusive_Thrombosis", "pathophysiology", "pathophysiology")
    ont.add_triple("Acute_Occlusive_Thrombosis", "caused_by", "Transmural_Myocardial_Ischemia", "pathophysiology", "pathophysiology")
    ont.add_triple("Transmural_Myocardial_Ischemia", "caused_by", "Myocyte_Coagulative_Necrosis", "pathophysiology", "pathophysiology")
    ont.add_triple("Myocyte_Coagulative_Necrosis", "caused_by", "Elevated_Cardiac_Troponin_I", "pathophysiology", "biomarker")
    ont.add_triple("Acute_STEMI", "treated_by", "Primary_Percutaneous_Coronary_Intervention", "disease", "drug")
    ont.add_triple("Acute_STEMI", "treated_by", "Aspirin_Dual_Antiplatelet", "disease", "drug")
    ont.add_triple("Acute_STEMI", "treated_by", "Unfractionated_Heparin", "disease", "drug")
    ont.add_triple("Acute_STEMI", "treated_by", "High_Intensity_Atorvastatin", "disease", "drug")

    # Acute Pericarditis
    ont.add_triple("Acute_Pericarditis", "presents_with", "Sharp_Pleuritic_Chest_Pain", "disease", "symptom")
    ont.add_triple("Acute_Pericarditis", "presents_with", "Pain_Relieved_Sitting_Forward", "disease", "symptom")
    ont.add_triple("Acute_Pericarditis", "presents_with", "Pericardial_Friction_Rub", "disease", "symptom")
    ont.add_triple("Acute_Pericarditis", "indicated_biomarker", "Diffuse_Concave_ST_Elevation", "disease", "biomarker")
    ont.add_triple("Acute_Pericarditis", "indicated_biomarker", "PR_Segment_Depression", "disease", "biomarker")
    ont.add_triple("Acute_Pericarditis", "caused_by", "Pericardial_Serofibrinous_Inflammation", "disease", "pathophysiology")
    ont.add_triple("Acute_Pericarditis", "treated_by", "Colchicine", "disease", "drug")
    ont.add_triple("Acute_Pericarditis", "treated_by", "High_Dose_Ibuprofen", "disease", "drug")

    # Aortic Dissection (Stanford Type A / B)
    ont.add_triple("Aortic_Dissection", "presents_with", "Tearing_Interscapular_Back_Pain", "disease", "symptom")
    ont.add_triple("Aortic_Dissection", "presents_with", "Pulse_Deficit_Between_Arms", "disease", "symptom")
    ont.add_triple("Aortic_Dissection", "presents_with", "Severe_Refractory_Hypertension", "disease", "symptom")
    ont.add_triple("Aortic_Dissection", "indicated_biomarker", "Mediastinal_Widening_on_CXR", "disease", "biomarker")
    ont.add_triple("Aortic_Dissection", "indicated_biomarker", "Intimal_Tear_on_CT_Angiography", "disease", "biomarker")
    ont.add_triple("Aortic_Dissection", "caused_by", "Cystic_Medial_Necrosis_and_Intimal_Tear", "disease", "pathophysiology")
    ont.add_triple("Aortic_Dissection", "treated_by", "Intravenous_Esmolol_Beta_Blockade", "disease", "drug")
    ont.add_triple("Aortic_Dissection", "treated_by", "Emergency_Surgical_Graft_Repair", "disease", "drug")
    # Contraindication
    ont.add_triple("Thrombolysis_tPA", "contraindicated_in", "Aortic_Dissection", "drug", "disease")

    # =========================================================================
    # 2. PULMONOLOGY
    # =========================================================================
    # Acute Pulmonary Embolism
    ont.add_triple("Pulmonary_Embolism", "presents_with", "Sudden_Onset_Dyspnea", "disease", "symptom")
    ont.add_triple("Pulmonary_Embolism", "presents_with", "Pleuritic_Chest_Pain", "disease", "symptom")
    ont.add_triple("Pulmonary_Embolism", "presents_with", "Hemoptysis", "disease", "symptom")
    ont.add_triple("Pulmonary_Embolism", "presents_with", "Tachycardia", "disease", "symptom")
    ont.add_triple("Pulmonary_Embolism", "presents_with", "Unilateral_Calf_Swelling_DVT", "disease", "symptom")
    ont.add_triple("Pulmonary_Embolism", "indicated_biomarker", "Markedly_Elevated_D_Dimer", "disease", "biomarker")
    ont.add_triple("Pulmonary_Embolism", "indicated_biomarker", "CT_Pulmonary_Angiogram_Filling_Defect", "disease", "biomarker")
    ont.add_triple("Pulmonary_Embolism", "indicated_biomarker", "S1Q3T3_Pattern_on_ECG", "disease", "biomarker")
    ont.add_triple("Deep_Vein_Thrombosis", "complication_of", "Pulmonary_Embolism", "disease", "disease")
    ont.add_triple("Pulmonary_Embolism", "caused_by", "Thrombus_Embolization_to_Pulmonary_Artery", "disease", "pathophysiology")
    ont.add_triple("Thrombus_Embolization_to_Pulmonary_Artery", "caused_by", "Ventilation_Perfusion_Mismatch", "pathophysiology", "pathophysiology")
    ont.add_triple("Ventilation_Perfusion_Mismatch", "caused_by", "Acute_Right_Ventricular_Strain", "pathophysiology", "pathophysiology")
    ont.add_triple("Pulmonary_Embolism", "treated_by", "Low_Molecular_Weight_Heparin", "disease", "drug")
    ont.add_triple("Pulmonary_Embolism", "treated_by", "Direct_Oral_Anticoagulants_Apixaban", "disease", "drug")
    ont.add_triple("Pulmonary_Embolism", "treated_by", "Catheter_Directed_Thrombolysis", "disease", "drug")

    # Bacterial Community-Acquired Pneumonia (CAP)
    ont.add_triple("Bacterial_Pneumonia", "presents_with", "Fever_with_Rigors", "disease", "symptom")
    ont.add_triple("Bacterial_Pneumonia", "presents_with", "Productive_Cough_with_Rusty_Sputum", "disease", "symptom")
    ont.add_triple("Bacterial_Pneumonia", "presents_with", "Bronchial_Breath_Sounds_and_Crackles", "disease", "symptom")
    ont.add_triple("Bacterial_Pneumonia", "indicated_biomarker", "Lobar_Consolidation_on_Chest_XRay", "disease", "biomarker")
    ont.add_triple("Bacterial_Pneumonia", "indicated_biomarker", "Elevated_C_Reactive_Protein_and_PCT", "disease", "biomarker")
    ont.add_triple("Bacterial_Pneumonia", "indicated_biomarker", "Neutrophilic_Leukocytosis", "disease", "biomarker")
    ont.add_triple("Bacterial_Pneumonia", "caused_by", "Streptococcus_Pneumoniae_Alveolar_Infiltration", "disease", "pathophysiology")
    ont.add_triple("Bacterial_Pneumonia", "treated_by", "Ceftriaxone_plus_Azithromycin", "disease", "drug")

    # Tension Pneumothorax
    ont.add_triple("Tension_Pneumothorax", "presents_with", "Hyperresonance_to_Percussion", "disease", "symptom")
    ont.add_triple("Tension_Pneumothorax", "presents_with", "Absent_Unilateral_Breath_Sounds", "disease", "symptom")
    ont.add_triple("Tension_Pneumothorax", "presents_with", "Tracheal_Deviation_to_Contralateral_Side", "disease", "symptom")
    ont.add_triple("Tension_Pneumothorax", "presents_with", "Severe_Hypotension_and_Shock", "disease", "symptom")
    ont.add_triple("Tension_Pneumothorax", "caused_by", "One_Way_Valve_Air_Entrapment", "disease", "pathophysiology")
    ont.add_triple("One_Way_Valve_Air_Entrapment", "caused_by", "Impaired_Venous_Return_Preload_Collapse", "pathophysiology", "pathophysiology")
    ont.add_triple("Tension_Pneumothorax", "treated_by", "Immediate_Needle_Decompression_2nd_ICS", "disease", "drug")
    ont.add_triple("Tension_Pneumothorax", "treated_by", "Tube_Thoracostomy_Intercostal_Drain", "disease", "drug")

    # =========================================================================
    # 3. GASTROENTEROLOGY & ACUTE SURGERY
    # =========================================================================
    # Acute Appendicitis
    ont.add_triple("Acute_Appendicitis", "presents_with", "Periumbilical_Pain_Migrating_to_RLQ", "disease", "symptom")
    ont.add_triple("Acute_Appendicitis", "presents_with", "Tenderness_at_McBurney_Point", "disease", "symptom")
    ont.add_triple("Acute_Appendicitis", "presents_with", "Positive_Rovsing_and_Psoas_Signs", "disease", "symptom")
    ont.add_triple("Acute_Appendicitis", "presents_with", "Low_Grade_Fever_and_Anorexia", "disease", "symptom")
    ont.add_triple("Acute_Appendicitis", "indicated_biomarker", "Non_Compressible_Appendix_USG_over_6mm", "disease", "biomarker")
    ont.add_triple("Acute_Appendicitis", "indicated_biomarker", "Target_Sign_on_Abdominal_CT", "disease", "biomarker")
    ont.add_triple("Acute_Appendicitis", "caused_by", "Appendiceal_Lumen_Obstruction_by_Fecalith", "disease", "pathophysiology")
    ont.add_triple("Appendiceal_Lumen_Obstruction_by_Fecalith", "caused_by", "Intraluminal_Bacterial_Overgrowth_and_Ischemia", "pathophysiology", "pathophysiology")
    ont.add_triple("Intraluminal_Bacterial_Overgrowth_and_Ischemia", "caused_by", "Gangrenous_Transmural_Perforation", "pathophysiology", "pathophysiology")
    ont.add_triple("Acute_Appendicitis", "treated_by", "Emergency_Laparoscopic_Appendectomy", "disease", "drug")
    ont.add_triple("Acute_Appendicitis", "treated_by", "Intravenous_Ceftriaxone_Metronidazole", "disease", "drug")

    # Acute Pancreatitis
    ont.add_triple("Acute_Pancreatitis", "presents_with", "Epigastric_Pain_Radiating_to_Back", "disease", "symptom")
    ont.add_triple("Acute_Pancreatitis", "presents_with", "Relief_on_Leaning_Forward", "disease", "symptom")
    ont.add_triple("Acute_Pancreatitis", "presents_with", "Nausea_and_Bilious_Vomiting", "disease", "symptom")
    ont.add_triple("Acute_Pancreatitis", "indicated_biomarker", "Serum_Lipase_Greater_Than_3x_Normal", "disease", "biomarker")
    ont.add_triple("Acute_Pancreatitis", "indicated_biomarker", "Peripancreatic_Fluid_Collections_on_CT", "disease", "biomarker")
    ont.add_triple("Acute_Pancreatitis", "caused_by", "Premature_Intraacinar_Trypsinogen_Activation", "disease", "pathophysiology")
    ont.add_triple("Premature_Intraacinar_Trypsinogen_Activation", "caused_by", "Autodigestion_and_Fat_Necrosis", "pathophysiology", "pathophysiology")
    ont.add_triple("Acute_Pancreatitis", "treated_by", "Aggressive_Intravenous_Ringer_Lactate", "disease", "drug")
    ont.add_triple("Acute_Pancreatitis", "treated_by", "Early_Enteral_Nutrition", "disease", "drug")

    # =========================================================================
    # 4. ENDOCRINOLOGY & METABOLISM
    # =========================================================================
    # Diabetic Ketoacidosis (DKA)
    ont.add_triple("Diabetic_Ketoacidosis", "presents_with", "Kussmaul_Deep_Rapid_Breathing", "disease", "symptom")
    ont.add_triple("Diabetic_Ketoacidosis", "presents_with", "Fruity_Acetone_Breath_Odor", "disease", "symptom")
    ont.add_triple("Diabetic_Ketoacidosis", "presents_with", "Polyuria_Polydipsia_Dehydration", "disease", "symptom")
    ont.add_triple("Diabetic_Ketoacidosis", "presents_with", "Abdominal_Pain_and_Altered_Sensorium", "disease", "symptom")
    ont.add_triple("Diabetic_Ketoacidosis", "indicated_biomarker", "Hyperglycemia_over_250mg_dL", "disease", "biomarker")
    ont.add_triple("Diabetic_Ketoacidosis", "indicated_biomarker", "High_Anion_Gap_Metabolic_Acidosis_pH_under_7_3", "disease", "biomarker")
    ont.add_triple("Diabetic_Ketoacidosis", "indicated_biomarker", "Positive_Serum_Beta_Hydroxybutyrate", "disease", "biomarker")
    ont.add_triple("Diabetic_Ketoacidosis", "caused_by", "Absolute_Insulin_Deficiency", "disease", "pathophysiology")
    ont.add_triple("Absolute_Insulin_Deficiency", "caused_by", "Unchecked_Lipolysis_and_Free_Fatty_Acids", "pathophysiology", "pathophysiology")
    ont.add_triple("Unchecked_Lipolysis_and_Free_Fatty_Acids", "caused_by", "Hepatic_Ketogenesis_Acetoacetate_BetaHB", "pathophysiology", "pathophysiology")
    ont.add_triple("Diabetic_Ketoacidosis", "treated_by", "Continuous_IV_Regular_Insulin_Infusion", "disease", "drug")
    ont.add_triple("Diabetic_Ketoacidosis", "treated_by", "Intravenous_Isotonic_0_9_Saline_Hydration", "disease", "drug")
    ont.add_triple("Diabetic_Ketoacidosis", "treated_by", "Potassium_Replacement_Protocol", "disease", "drug")

    # =========================================================================
    # 5. NEUROLOGY
    # =========================================================================
    # Acute Bacterial Meningitis
    ont.add_triple("Bacterial_Meningitis", "presents_with", "Fever_and_Chills", "disease", "symptom")
    ont.add_triple("Bacterial_Meningitis", "presents_with", "Severe_Throbbing_Headache", "disease", "symptom")
    ont.add_triple("Bacterial_Meningitis", "presents_with", "Nuchal_Rigidity_Neck_Stiffness", "disease", "symptom")
    ont.add_triple("Bacterial_Meningitis", "presents_with", "Positive_Kernig_and_Brudzinski_Signs", "disease", "symptom")
    ont.add_triple("Bacterial_Meningitis", "presents_with", "Photophobia", "disease", "symptom")
    ont.add_triple("Bacterial_Meningitis", "indicated_biomarker", "CSF_Neutrophilic_Pleocytosis", "disease", "biomarker")
    ont.add_triple("Bacterial_Meningitis", "indicated_biomarker", "Markedly_Decreased_CSF_Glucose_under_40mg", "disease", "biomarker")
    ont.add_triple("Bacterial_Meningitis", "indicated_biomarker", "Elevated_CSF_Protein_over_100mg", "disease", "biomarker")
    ont.add_triple("Bacterial_Meningitis", "caused_by", "Hematogenous_Seeding_to_Subarachnoid_Space", "disease", "pathophysiology")
    ont.add_triple("Hematogenous_Seeding_to_Subarachnoid_Space", "caused_by", "Meningeal_Neutrophilic_Inflammation_Exudate", "pathophysiology", "pathophysiology")
    ont.add_triple("Bacterial_Meningitis", "treated_by", "IV_Ceftriaxone_plus_Vancomycin", "disease", "drug")
    ont.add_triple("Bacterial_Meningitis", "treated_by", "Adjuvant_IV_Dexamethasone", "disease", "drug")

    # Acute Ischemic Stroke (MCA)
    ont.add_triple("Acute_Ischemic_Stroke_MCA", "presents_with", "Contralateral_Hemiparesis_Face_Arm_over_Leg", "disease", "symptom")
    ont.add_triple("Acute_Ischemic_Stroke_MCA", "presents_with", "Expressive_or_Receptive_Aphasia", "disease", "symptom")
    ont.add_triple("Acute_Ischemic_Stroke_MCA", "presents_with", "Facial_Droop", "disease", "symptom")
    ont.add_triple("Acute_Ischemic_Stroke_MCA", "presents_with", "Dysarthria_Slurred_Speech", "disease", "symptom")
    ont.add_triple("Acute_Ischemic_Stroke_MCA", "indicated_biomarker", "Diffusion_Restriction_on_Brain_MRI", "disease", "biomarker")
    ont.add_triple("Acute_Ischemic_Stroke_MCA", "indicated_biomarker", "Hyperdense_MCA_Sign_on_NonContrast_CT", "disease", "biomarker")
    ont.add_triple("Acute_Ischemic_Stroke_MCA", "caused_by", "Thromboembolic_Occlusion_of_Middle_Cerebral_Artery", "disease", "pathophysiology")
    ont.add_triple("Thromboembolic_Occlusion_of_Middle_Cerebral_Artery", "caused_by", "Ischemic_Penumbra_and_Cytotoxic_Edema", "pathophysiology", "pathophysiology")
    ont.add_triple("Acute_Ischemic_Stroke_MCA", "treated_by", "Intravenous_Alteplase_tPA_under_4_5h", "disease", "drug")
    ont.add_triple("Acute_Ischemic_Stroke_MCA", "treated_by", "Endovascular_Mechanical_Thrombectomy", "disease", "drug")

    return ont
