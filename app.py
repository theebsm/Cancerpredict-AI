
import streamlit as st
import pickle
import numpy as np
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, QED

st.set_page_config(
    page_title="CancerPredict AI",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1a2e 100%); }
    h1 { background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6b6b);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-size: 3rem !important; font-weight: 900 !important; }
    h2, h3 { color: #00d4ff !important; }
    .stButton > button {
        background: linear-gradient(90deg, #7b2ff7, #00d4ff) !important;
        color: white !important; border: none !important;
        border-radius: 25px !important; padding: 10px 25px !important;
        font-weight: bold !important; font-size: 16px !important; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #1a1a3e !important; color: #00d4ff !important;
        border: 2px solid #7b2ff7 !important; border-radius: 10px !important; }
    .stMetric { background: linear-gradient(135deg, #1a1a3e, #2a1a4e) !important;
        border: 1px solid #7b2ff7 !important; border-radius: 15px !important;
        padding: 15px !important; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1a3e !important;
        color: #00d4ff !important; border-radius: 10px !important;
        margin-right: 5px !important; font-weight: bold !important; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #7b2ff7, #00d4ff) !important;
        color: white !important; }
    .stProgress > div > div { background: linear-gradient(90deg, #7b2ff7, #00d4ff) !important; }
    p, li { color: #c0c0e0 !important; }
    label { color: #00d4ff !important; font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)

# ── Drug SMILES Database ─────────────────────────────────────
DRUG_DB = {
    "imatinib"        : "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "gefitinib"       : "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "erlotinib"       : "COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC",
    "sorafenib"       : "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "dasatinib"       : "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
    "nilotinib"       : "Cc1ccc(-c2cn(C)c3cc(NC(=O)c4ccc(C)c(Nc5nccc(-c6cccnc6)n5)c4)ccc23)cc1",
    "sunitinib"       : "CCN(CC)CCNC(=O)c1c(C)[nH]c(C=C2C(=O)Nc3ccc(F)cc32)c1C",
    "lapatinib"       : "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "vemurafenib"     : "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ccc(Cl)cc23)c1",
    "crizotinib"      : "Cc1cn(C2CCNC2)c2cc(Nc3ccc(F)cc3Cl)c(Cl)cc12",
    "tamoxifen"       : "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "paclitaxel"      : "O=C(O[C@@H]1C[C@]2(O)C(=O)[C@@H](OC(=O)c3ccccc3)c3cc(OC(C)=O)ccc3[C@@H]2[C@H]1OC(=O)c1ccccc1)c1ccccc1",
    "doxorubicin"     : "O=C1c2cccc(O)c2C(=O)c2c(O)c3c(c(O)c21)C[C@@](O)(C(=O)CO)C3",
    "cisplatin"       : "[NH3][Pt](Cl)(Cl)[NH3]",
    "carboplatin"     : "O=C1OC(=O)C12CC[Pt](N)(N)CC2",
    "methotrexate"    : "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
    "cyclophosphamide": "ClCCNP1(=O)OCCCN1CCCl",
    "temozolomide"    : "Cn1nnc2c(C(N)=O)ncn2c1=O",
    "bortezomib"      : "CC(C)C[C@@H](NC(=O)[C@@H](Cc1cccnc1)NC(=O)c1cnccn1)B(O)O",
    "vincristine"     : "CCC1(CC(CC2(C1NC3=CC=CC=C23)C(=O)OC)N4CCC5=C(C4)NC6=CC=CC=C56)O",
    "gemcitabine"     : "NC(=O)c1ccn(C2OC(CO)C(O)C2(F)F)c(=O)n1",
    "capecitabine"    : "CCCCOC(=O)Nc1nc(=O)n(C2OC(C)C(O)C2O)cc1F",
    "oxaliplatin"     : "O=C1OC(=O)[C@@H]2CCCC[C@H]2N1[Pt]",
    "lenalidomide"    : "NC1=CC2=C(C=C1)C(=O)N(C1CCC(=O)NC1=O)C2=O",
    "topotecan"       : "OC(=O)c1cn2cc3c(cc2c(=O)c1CC)C(=O)N1CC(O)(CC)Cc2cc3ccc21",
    "aspirin"         : "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine"        : "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen"       : "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "paracetamol"     : "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen"   : "CC(=O)Nc1ccc(O)cc1",
    "metformin"       : "CN(C)C(=N)NC(N)=N",
    "amoxicillin"     : "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O",
    "morphine"        : "OC1=CC=C2CC3N(C)CCC34C2=C1OC4",
    "diazepam"        : "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "warfarin"        : "OC(=O)c1ccccc1OCC(=O)Cc1c(O)c2ccccc2oc1=O",
    "dopamine"        : "NCCc1ccc(O)c(O)c1",
    "serotonin"       : "NCCc1c[nH]c2ccc(O)cc12",
    "melatonin"       : "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
    "glucose"         : "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "folic acid"      : "Nc1nc2ncc(CNc3ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc3)nc2c(=O)[nH]1",
    "vitamin c"       : "OCC(O)C1OC(=O)C(O)=C1O",
    "riboflavin"      : "Cc1cc2nc3c(=O)[nH]c(=O)nc3n(CC(O)C(O)C(O)CO)c2cc1C",
    "biotin"          : "OC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12",
    "atorvastatin"    : "CC(C)c1n(CC(O)CC(O)CC(=O)O)c(-c2ccc(F)cc2)c(C(=O)Nc2ccccc2)c1C(=O)O",
    "omeprazole"      : "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "osimertinib"     : "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1N(C)CCN(C)C",
    "palbociclib"     : "CC1=C(C(=O)Nc2ncnc3[nH]ccc23)C=CN1C1CCCC1",
    "ibrutinib"       : "C=CC(=O)N1CCC[C@@H](n2nc(-c3ccc(Oc4ccccc4)cc3)c3c(N)ncnc32)C1",
    "venetoclax"      : "CC1(CCC(=C2C=CC(=CC2=O)c2ccc(Cl)cc2)CC1)c1ccc(NC(=O)c2ccc(N3CCN(CC3)c3ccccc3)cc2)cc1",
}

def lookup_smiles(drug_name):
    # Step 1: local database
    name_lower = drug_name.lower().strip()
    if name_lower in DRUG_DB:
        return DRUG_DB[name_lower], "✅ Found in local database"

    # Step 2: try PubChem
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/IsomericSMILES/JSON"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            smiles = r.json()["PropertyTable"]["Properties"][0]["IsomericSMILES"]
            return smiles, "✅ Found on PubChem"
    except:
        pass

    # Step 3: try NCI CACTUS
    try:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{drug_name}/smiles"
        r = requests.get(url, timeout=8)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip(), "✅ Found on NCI CACTUS"
    except:
        pass

    return None, "❌ Not found — please paste SMILES manually"

@st.cache_resource
def load_models():
    with open("/content/cancerpredict_app/models/random_forest.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("/content/cancerpredict_app/models/xgboost.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open("/content/cancerpredict_app/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return rf, xgb, scaler

rf_model, xgb_model, scaler = load_models()

def smiles_to_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp     = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        morgan = np.array(fp)
        props  = np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            Descriptors.TPSA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol)
        ])
        return np.concatenate([morgan, props])
    except:
        return None

# ── Header ───────────────────────────────────────────────────
st.markdown("<h1>🧬 CancerPredict AI</h1>", unsafe_allow_html=True)
st.markdown("### 🔬 AI-Powered Anticancer Drug Discovery Platform")
st.markdown("💊 Trained on **10,893 real ChEMBL compounds** &nbsp;|&nbsp; 🎯 Accuracy: **87.98%** &nbsp;|&nbsp; 📊 AUC: **0.93**")
st.divider()

# ── Input Section ────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 💉 Drug Input")
    drug_name = st.text_input("Drug Name", placeholder="e.g. Imatinib, Sorafenib...")

    # Auto lookup button
    if st.button("🔍 Auto-Generate SMILES from Name", use_container_width=True):
        if drug_name:
            with st.spinner(f"Looking up {drug_name}..."):
                found_smiles, source = lookup_smiles(drug_name)
            if found_smiles:
                st.session_state["smiles_value"] = found_smiles
                st.session_state["smiles_source"] = source
                st.success(source)
            else:
                st.error(source)
                st.info("💡 Tip: Paste the SMILES manually from PubChem.ncbi.nlm.nih.gov")
        else:
            st.warning("Please enter a drug name first!")

    # Show source if found
    if "smiles_source" in st.session_state:
        st.info(st.session_state["smiles_source"])

    smiles = st.text_area(
        "SMILES String",
        value=st.session_state.get("smiles_value", ""),
        height=120,
        placeholder="Click Auto-Generate above OR paste SMILES here..."
    )

    with st.expander("📋 50+ Supported Drugs"):
        drugs_list = ", ".join([d.title() for d in DRUG_DB.keys()])
        st.write(drugs_list)

    analyze = st.button("🔬 Analyze Drug", type="primary", use_container_width=True)

if analyze and drug_name and smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES! Please check the structure.")
    else:
        with col2:
            st.markdown(f"### 📊 Results: **{drug_name}**")

            features = smiles_to_features(smiles)
            features_scaled = scaler.transform([features])

            rf_pred  = rf_model.predict(features_scaled)[0]
            rf_prob  = rf_model.predict_proba(features_scaled)[0][1]
            xgb_pred = xgb_model.predict(features_scaled)[0]
            xgb_prob = xgb_model.predict_proba(features_scaled)[0][1]
            avg_prob = (rf_prob + xgb_prob) / 2

            if avg_prob >= 0.5:
                st.success(f"✅ ACTIVE ANTICANCER COMPOUND — {avg_prob*100:.1f}% Confidence")
            else:
                st.error(f"❌ NOT ACTIVE — {avg_prob*100:.1f}% Confidence")
            st.progress(float(avg_prob))

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🤖 ML Prediction", "☣️ Toxicity", "⚗️ Properties",
                "🎯 Targets", "♻️ Repurposing"
            ])

            with tab1:
                c1, c2, c3 = st.columns(3)
                c1.metric("🌲 Random Forest", f"{rf_prob*100:.1f}%",
                           "ACTIVE" if rf_pred == 1 else "NOT ACTIVE")
                c2.metric("⚡ XGBoost", f"{xgb_prob*100:.1f}%",
                           "ACTIVE" if xgb_pred == 1 else "NOT ACTIVE")
                c3.metric("📊 Avg Confidence", f"{avg_prob*100:.1f}%")
                if avg_prob >= 0.8:
                    st.success("🏆 HIGH CONFIDENCE — Strong anticancer candidate!")
                elif avg_prob >= 0.5:
                    st.warning("⚠️ MODERATE CONFIDENCE — Potential anticancer activity")
                else:
                    st.error("❌ LOW CONFIDENCE — Unlikely to be anticancer")

            with tab2:
                mw   = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
                rings= rdMolDescriptors.CalcNumRings(mol)
                qed  = QED.qed(mol)
                flags = []
                if logp > 3 and mw < 300: flags.append("Pfizer 3/75 rule")
                if logp > 5             : flags.append("High logP")
                if mw > 500             : flags.append("High MW")
                if rings > 6            : flags.append("Many Rings")
                if rotb > 10            : flags.append("Many RotBonds")
                if tpsa < 20            : flags.append("Low TPSA")
                risk = "LOW" if len(flags) == 0 else "MODERATE" if len(flags) <= 2 else "HIGH"
                c1, c2 = st.columns(2)
                c1.metric("💊 QED Score", f"{qed:.3f} / 1.0")
                c2.metric("☣️ Risk Level", risk)
                if risk == "LOW":
                    st.success("✅ LOW TOXICITY RISK")
                elif risk == "MODERATE":
                    st.warning(f"⚠️ MODERATE RISK — {', '.join(flags)}")
                else:
                    st.error(f"❌ HIGH RISK — {', '.join(flags)}")

            with tab3:
                mw   = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
                rings= rdMolDescriptors.CalcNumRings(mol)
                qed  = QED.qed(mol)
                hbd  = rdMolDescriptors.CalcNumHBD(mol)
                hba  = rdMolDescriptors.CalcNumHBA(mol)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("⚖️ MW",    f"{mw:.1f} Da")
                c2.metric("💧 logP",  f"{logp:.2f}")
                c3.metric("🔬 TPSA",  f"{tpsa:.1f}")
                c4.metric("⭐ QED",   f"{qed:.3f}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🔗 HBD",   hbd)
                c2.metric("🔗 HBA",   hba)
                c3.metric("🔄 RotB",  rotb)
                c4.metric("💍 Rings", rings)
                viols = sum([mw >= 500, logp >= 5, hbd >= 5, hba >= 10])
                veber = rotb <= 10 and tpsa <= 140
                col_a, col_b = st.columns(2)
                with col_a:
                    if viols == 0:
                        st.success("✅ Lipinski: PASS")
                    else:
                        st.warning(f"⚠️ Lipinski: {viols} violation(s)")
                with col_b:
                    if veber:
                        st.success("✅ Veber Rules: PASS")
                    else:
                        st.warning("⚠️ Veber Rules: FAIL")

            with tab4:
                target_refs = {
                    "🎯 EGFR"        : "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
                    "🎯 BCR-ABL"     : "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
                    "🎯 VEGFR2"      : "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
                    "🎯 COX"         : "CC(=O)Oc1ccccc1C(=O)O",
                    "🎯 Topoisomerase": "COc1cc2c(cc1OC)c1c3c(c(=O)n2CCN(C)C)cccc3c(=O)c1",
                    "🎯 Tubulin"     : "CC1=C2C=C(C=CC2=NC=C1)OC",
                }
                query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                sims = []
                for target, ref_smi in target_refs.items():
                    ref_mol = Chem.MolFromSmiles(ref_smi)
                    if ref_mol is None: continue
                    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, 2048)
                    sims.append((target, DataStructs.TanimotoSimilarity(query_fp, ref_fp)))
                sims.sort(key=lambda x: x[1], reverse=True)
                for target, sim in sims:
                    conf = "🔴 HIGH" if sim > 0.5 else "🟡 MODERATE" if sim > 0.3 else "🟢 LOW"
                    col_t, col_s, col_c = st.columns([3,1,1])
                    col_t.write(f"**{target}**")
                    col_s.write(f"`{sim:.3f}`")
                    col_c.write(conf)
                    st.progress(float(sim))

            with tab5:
                approved = {
                    "Imatinib"   : ("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "BCR-ABL", "CML/Leukemia"),
                    "Gefitinib"  : ("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",               "EGFR",    "Lung Cancer"),
                    "Sorafenib"  : ("CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",   "VEGFR/RAF","Kidney/Liver Cancer"),
                    "Erlotinib"  : ("COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OC",                            "EGFR",    "Lung/Pancreatic Cancer"),
                    "Dasatinib"  : ("Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",      "BCR-ABL/SRC","CML/ALL Cancer"),
                    "Tamoxifen"  : ("CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",                   "Estrogen Receptor","Breast Cancer"),
                    "Paclitaxel" : ("O=C(O[C@@H]1C[C@]2(O)C(=O)[C@@H](OC(=O)c3ccccc3)c3cc(OC(C)=O)ccc3[C@@H]2[C@H]1OC(=O)c1ccccc1)c1ccccc1","Tubulin","Breast/Ovarian Cancer"),
                    "Doxorubicin": ("O=C1c2cccc(O)c2C(=O)c2c(O)c3c(c(O)c21)C[C@@](O)(C(=O)CO)C3",   "Topoisomerase II","Breast/Ovarian Cancer"),
                }
                rep_sims = []
                for dname, (dsmi, dtarget, dind) in approved.items():
                    if dname.lower() == drug_name.lower(): continue
                    dmol = Chem.MolFromSmiles(dsmi)
                    if dmol is None: continue
                    dfp  = AllChem.GetMorganFingerprintAsBitVect(dmol, 2, 2048)
                    sim  = DataStructs.TanimotoSimilarity(query_fp, dfp)
                    rep_sims.append((dname, sim, dtarget, dind))
                rep_sims.sort(key=lambda x: x[1], reverse=True)
                for dname, sim, dtarget, dind in rep_sims[:5]:
                    emoji = "🥇" if sim > 0.3 else "🥈" if sim > 0.2 else "🥉"
                    with st.expander(f"{emoji} {dname} — {sim*100:.1f}% similarity | {dind}"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("💊 Drug", dname)
                        c2.metric("🎯 Target", dtarget)
                        c3.metric("📊 Similarity", f"{sim*100:.1f}%")
                        st.write(f"**Indication:** {dind}")
                        st.progress(float(sim))
                        if sim > 0.3:
                            st.success(f"✅ HIGH similarity — {drug_name} may work for {dind}")
                        elif sim > 0.15:
                            st.warning(f"⚠️ MODERATE — worth investigating for {dind}")

            st.divider()
            score = 0
            if avg_prob >= 0.5           : score += 3
            if risk == "LOW"             : score += 2
            if viols <= 1                : score += 2
            if rotb <= 10 and tpsa <= 140: score += 2
            if qed > 0.5                 : score += 1
            overall = "🏆 EXCELLENT" if score >= 8 else "✅ GOOD" if score >= 6 else "⚠️ MODERATE" if score >= 4 else "❌ POOR"
            st.markdown(f"## Overall Score: `{score}/10` — {overall}")

elif analyze:
    st.warning("⚠️ Please enter a drug name and SMILES string.")
