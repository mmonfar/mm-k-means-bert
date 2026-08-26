"""
data_generator.py
mmonfar. // Semantic M&M Failure Navigation Engine — Phase 3.1

Generates `mock_mm_minutes.xlsx`: exactly 100 rows of synthetic Mortality & Morbidity
meeting log summaries written the way real minutes are actually written — clipped,
jargon-heavy, inconsistent, and full of shorthand.

The corpus is deliberately adversarial. Baked-in semantic traps:

  1. LEXICAL DIVERGENCE, SEMANTIC IDENTITY
     "blood thinner mistake" / "heparin administration error" / "warfarin dose
     miscalculated" share no keywords but are one failure mode. A keyword pivot
     table scatters them; an embedding model should pull them together.

  2. LEXICAL OVERLAP, SEMANTIC DIVERGENCE
     "handover of the ventilator circuit to biomed" (equipment) vs "ventilator
     settings not handed over at shift change" (communication). Same nouns,
     different failure.

  3. NEGATION LOOPS
     "No surgical complication noted; delay was in getting to theatre." A bag-of-words
     model files this under surgical complications. It is a delay case.

  4. DEPARTMENT DECOYS
     Department is uncorrelated-by-design with failure mode, so any clustering that
     "works" by leaking the department label is immediately visible as wrong.

NO PHI. Every name, date, and case is invented.

Usage:
    python data_generator.py
    python data_generator.py --rows 100 --seed 42 --out mock_mm_minutes.xlsx
"""

from __future__ import annotations

import argparse
import datetime as dt
import random

import pandas as pd

# --------------------------------------------------------------------------------------
# Ground-truth failure modes. Stored only as a generation scaffold — the label is NOT
# written to the spreadsheet, because the whole point is that the engine must recover
# this structure from the prose alone.
# --------------------------------------------------------------------------------------

MEDICATION_ERRORS = [
    "Pt received 10x intended dose of IV morphine overnight; decimal point error on the "
    "handwritten drug chart, not caught at second check. Naloxone given, full recovery.",
    "Blood thinner mistake on the ward — pt was on apixaban at home, restarted post-op "
    "without holding for the epidural. Epidural haematoma, urgent MRI.",
    "Heparin administration error: infusion running at 1400 units/hr instead of 800. "
    "APTT supratherapeutic at 6h. Pump programmed from the wrong protocol card.",
    "Warfarin dose miscalculated on discharge TTO. INR 8.2 at community check 4 days later. "
    "No bleeding but readmitted for vitamin K.",
    "Gentamicin dosed on actual body weight in a morbidly obese pt. AKI stage 2 by day 3. "
    "Pharmacy flagged it but the alert sat unactioned in the EPR queue.",
    "Pt with documented penicillin anaphylaxis given co-amoxiclav in ED. Allergy band present "
    "but allergy field in EPR was free-text, not coded, so no hard stop fired.",
    "Insulin units written as 'u' — read as a zero. 8u became 80u. Severe hypo, GCS 6, "
    "recovered with IV dextrose. Abbreviation banned locally since 2019.",
    "Double-dosing of paracetamol: regular IV prescribed on the anaesthetic chart and oral "
    "PRN on the ward chart simultaneously. Two charts never reconciled.",
    "Chemo cycle given at day 14 instead of day 21. Neutropenic sepsis. Cycle interval "
    "transcribed incorrectly onto the local schedule from the trial protocol.",
    "Potassium infusion given peripherally at 40mmol/L concentration. Severe phlebitis and "
    "extravasation injury. Central access was available and unused.",
    "Wrong-patient prescribing — two pts with same surname on the same bay, drug chart opened "
    "on the wrong record. Antihypertensive given to a normotensive pt, symptomatic hypotension.",
    "Methotrexate prescribed daily rather than weekly on discharge summary. Community pharmacy "
    "queried it; the query went to a shared inbox nobody owned. Pancytopenia at 2 weeks.",
    "Anticoagulation not restarted after endoscopy despite plan documented in the procedure "
    "note. Pt had a mechanical valve. Embolic CVA on day 5.",
    "Opioid-naive pt started on long-acting oxycodone plus PRN immediate release with no "
    "review interval set. Respiratory rate 6 at 0400, found by HCA on obs round.",
    "Enoxaparin prophylaxis omitted for 4 consecutive days — 'nil by mouth' status was "
    "wrongly interpreted as hold-all-meds by the covering team. Subsequent PE.",
    "Vancomycin levels not taken before the third dose. Trough 32. Ototoxicity screening "
    "arranged. Level request had been raised but never actioned by phlebotomy.",
    "Sound-alike substitution: hydralazine dispensed instead of hydroxyzine from ward stock. "
    "Both stored in adjacent bins with near-identical box livery.",
    "Pt self-administered their own home inhalers alongside ward-prescribed nebulisers; "
    "cumulative salbutamol load, tachyarrhythmia. No medicines reconciliation on admission.",
    "IV fluid prescribed as 0.9% saline 3L/24h in a pt with known heart failure and EF 25%. "
    "Flash pulmonary oedema overnight. Fluid chart not reviewed at the post-take round.",
    "Contrast given despite documented severe iodinated contrast reaction. The reaction was "
    "recorded in a scanned letter, not in the structured allergy list.",
]

SURGICAL_COMPLICATIONS = [
    "Anastomotic leak POD4 following anterior resection. Return to theatre, defunctioning "
    "stoma formed. Leak rate for this procedure remains within unit benchmark.",
    "Retained swab identified on post-op CXR after emergency laparotomy. Count had been "
    "declared correct; count was performed during an unplanned staff swap mid-case.",
    "Bile duct injury during difficult lap chole — Strasberg E2. Converted to open, "
    "hepatobiliary referral same day. Critical view of safety not achieved before clipping.",
    "Intraoperative haemorrhage from splenic capsular tear during a left hemicolectomy. "
    "4 units transfused. Splenectomy avoided with haemostatic agent.",
    "Wrong-level spinal surgery — decompression at L3/4 rather than L4/5. Intraoperative "
    "imaging performed but counted from the wrong reference vertebra.",
    "Deep sternal wound infection post CABG in a poorly-controlled diabetic. VAC dressing, "
    "pectoral flap at week 3. HbA1c 94 pre-op, surgery not deferred.",
    "Ureteric injury during total abdominal hysterectomy for a large fibroid uterus. "
    "Recognised intraoperatively, urology stented. Pre-op imaging had shown deviation.",
    "Post-tonsillectomy secondary haemorrhage day 7, returned to theatre for arrest of "
    "bleeding. Airway compromised on induction, difficult intubation drill activated.",
    "Bowel perforation on trocar insertion in a pt with multiple prior laparotomies. Open "
    "entry technique had been discussed and not used.",
    "Compartment syndrome of the leg missed for 11 hours post tibial nailing. Fasciotomy "
    "performed late; residual foot drop at 6 months.",
    "Prosthetic joint infection at 5 weeks post primary TKR. Debridement and liner exchange. "
    "Prophylactic antibiotic given after tourniquet inflation rather than before.",
    "Anaesthetic complication — dental damage on laryngoscopy in a pt with known loose "
    "crowns. Documented in pre-assessment, not carried into the theatre brief.",
    "Vascular graft thrombosis at 48h post fem-pop bypass. Urgent thrombectomy. Antiplatelet "
    "had been withheld for an unrelated planned dental extraction.",
    "Iatrogenic pneumothorax following supraclavicular block, chest drain required. "
    "Ultrasound guidance in use but needle tip lost from view during advancement.",
    "Massive transfusion triggered in a placenta accreta case where accreta was not "
    "anticipated antenatally. Hysterectomy performed, mother survived.",
    "Nerve injury — common peroneal palsy following prolonged lithotomy positioning for a "
    "6-hour pelvic case. Positioning checks not repeated after the 2-hour mark.",
    "Post-op collection following appendicectomy in a perforated case, drained radiologically "
    "on day 6. Recognised course rather than a departure from standard.",
    "Skin flap necrosis after mastectomy in an active smoker. Smoking cessation was offered "
    "and declined; documented consent discussion covered this risk.",
    "Failed extubation post thyroidectomy from bilateral recurrent laryngeal nerve palsy. "
    "Emergency tracheostomy. Nerve monitoring was unavailable that list.",
    "Instrument fracture — tip of a laparoscopic grasper sheared and retained in the abdomen, "
    "retrieved same sitting under vision. Instrument was overdue for service.",
]

COMMUNICATION_HANDOFF = [
    "Night handover was verbal only, no written list. Pt awaiting urgent CT head was not "
    "mentioned. Scan happened 9 hours later than intended.",
    "Ventilator settings not handed over at shift change — new nurse assumed pressure support "
    "was the standing mode. Pt on mandatory ventilation, asynchrony and agitation for 2h.",
    "Critical potassium of 6.9 phoned to the ward by lab; the message was taken by a ward "
    "clerk and written on a sticky note. Never reached a clinician.",
    "Escalation failure — F1 concerned about a deteriorating pt, bleeped registrar twice, "
    "no response, did not escalate further because of perceived hierarchy.",
    "Transfer from ICU to ward with no medical handover, only nursing. Steroid taper plan "
    "and adrenal insufficiency risk not conveyed. Addisonian crisis on day 2.",
    "Consultant plan documented in the notes as 'for theatre if deteriorates' with no "
    "parameters defined. Overnight team had no threshold to act on.",
    "Radiology reported an incidental lung nodule; report auto-filed to a clinician who had "
    "left the trust. No safety-netting loop. Picked up 14 months later.",
    "Language barrier — consent for a procedure taken via the pt's adult son rather than an "
    "interpreter. Post-op the pt stated they had not understood the stoma risk.",
    "Two teams (medical and surgical) both documented in the notes, neither acknowledged the "
    "other's plan. Contradictory fluid instructions on the same day.",
    "Weekend cross-cover received a list of 46 pts with no priority ordering and no clinical "
    "context. The sickest pt was reviewed fifth.",
    "Handoff of an unwell pt occurred in a corridor during a fire alarm. No structured tool "
    "used. Allergy status and resus decision both omitted.",
    "DNACPR decision made and discussed with family but not communicated to the ambulance "
    "crew on transfer. CPR attempted inappropriately during conveyance.",
    "GP referral letter contained the key red-flag history; the letter was scanned into the "
    "record but ED triage worked from the verbal account only.",
    "Microbiology advice given by phone to a junior, not documented anywhere. The advice was "
    "then contradicted by the next day's team, antibiotic changed twice.",
    "Pt moved bed four times in 36 hours for flow reasons. The team lost track; ward round "
    "missed them entirely on day 2.",
    "Discharge summary sent to the GP with the medication changes section left as the default "
    "template text. Community team continued the pre-admission regimen.",
    "Theatre list order changed without informing the ward. Pt not starved appropriately, "
    "case cancelled, re-listed 5 days later with disease progression in the interval.",
    "Nursing concern raised at 0200 was recorded in the nursing notes only; medical notes and "
    "nursing notes are separate systems here and are not cross-read.",
    "SBAR not used on a deteriorating-patient call — the referral led with the social history "
    "and the news score was mentioned last. Response was not prioritised.",
    "Family raised a concern about reduced responsiveness directly to a passing porter; the "
    "concern never entered any clinical record. No Martha's-Rule style route existed.",
]

EQUIPMENT_DEVICE_FAILURE = [
    "Infusion pump free-flowed when the cassette was reseated; 200ml of noradrenaline "
    "delivered as a bolus. Device quarantined and reported to MHRA.",
    "Defibrillator failed self-test at the start of the arrest call. Backup unit fetched from "
    "the adjacent ward, 3-minute delay to first shock.",
    "Handover of the ventilator circuit to biomed for servicing left the unit one machine "
    "short overnight; a second admission had to be diverted.",
    "Piped oxygen outlet in bay 4 delivering below expected pressure. Estates confirmed a "
    "partially closed isolation valve after a weekend refurbishment.",
    "Central venous catheter guidewire sheared on withdrawal through the introducer needle. "
    "Fragment retrieved by interventional radiology.",
    "Blood gas analyser gave three implausible lactates in succession; cartridge had expired. "
    "Expiry checks were on a paper log that had lapsed.",
    "Telemetry pack battery failed silently — no alarm at the central station. Pt had an "
    "unwitnessed run of VT discovered later on the implanted device interrogation.",
    "Warming blanket thermostat fault caused a contact burn on the flank during a long case. "
    "Device had passed PPM two weeks earlier.",
    "Endoscope reprocessing cycle aborted mid-run; the scope was returned to the rack as clean. "
    "Traceability audit triggered recall of 6 pts for testing.",
    "Surgical diathermy return electrode partially detached, causing a full-thickness burn at "
    "the pad site. Contact quality monitor was disabled on that generator model.",
    "Syringe driver keypad unresponsive after a cleaning-fluid ingress. Analgesia interrupted "
    "for 90 minutes on a palliative pt.",
    "EPR outage for 4 hours with no printed downtime pack available on the ward. Observations "
    "recorded on paper towels and later transcribed with gaps.",
    "Pulse oximeter consistently over-reading in a pt with deeply pigmented skin; occult "
    "hypoxaemia. Known device-class limitation, no local guidance in place.",
    "Nasogastric tube pH paper stock had been discontinued by procurement without notifying "
    "wards. Placement confirmed by auscultation, a banned practice.",
    "Theatre lights failed during a laparoscopic case; generator switchover took 40 seconds. "
    "Case completed with headlamps, no harm.",
    "Patient trolley brake mechanism failed during transfer, trolley moved during a slide "
    "transfer. Staff back injury and a near-miss for the pt.",
    "Dialysis machine conductivity alarm ignored as it had cried wolf all week. The alarm was "
    "genuine; treatment aborted, no harm, machine withdrawn.",
    "Bair-hugger and forced-air units in short supply meant a case proceeded without warming; "
    "core temp 34.6 at end of surgery, prolonged recovery stay.",
    "Implant tray delivered with a mismatched trial component set. Case delayed 55 minutes "
    "under anaesthesia while the correct tray was couriered.",
    "Automated dispensing cabinet drawer misloaded by the restocking service — two strengths "
    "of the same drug in one pocket. Caught at the bedside check, near miss.",
]

DIAGNOSTIC_INTERVENTION_DELAY = [
    "No surgical complication noted; the delay was in getting to theatre. Perforated viscus "
    "waited 14 hours for an emergency slot because of list contention.",
    "Sepsis six not completed within the hour. Antibiotics given at 3h40 from triage. "
    "Recognition was prompt, the delay was entirely in drug availability and access.",
    "Missed subarachnoid — CT at 8 hours reported normal, LP not performed, discharged. "
    "Represented at day 3 with rebleed.",
    "Aortic dissection labelled as ACS for 6 hours. Chest pain pathway is highly optimised "
    "for ACS and pulls everything into it.",
    "Fractured neck of femur waited 62 hours for surgery against a 36-hour standard. Delay "
    "was anticoagulation reversal plus two consecutive list cancellations.",
    "Cauda equina symptoms present on the initial ED note; MRI requested as routine rather "
    "than emergency. Scanned 26 hours later, decompressed late.",
    "Suspected stroke arrived within window, but door-to-needle was 118 minutes because CT "
    "was occupied by a trauma call and no escalation route existed.",
    "Testicular torsion in a teenager attributed to epididymitis. Ultrasound at 9 hours. "
    "Orchidectomy at exploration.",
    "Deteriorating NEWS score of 9 recorded at 2200 and again at 2300; medical review not "
    "attended until 0130. No harm but avoidable delay to intervention.",
    "Positive blood culture flagged in the lab at 1400; result acknowledged at 0900 the next "
    "day. Endocarditis diagnosis delayed by a full day.",
    "Malignancy suspected on a CT performed for another indication. The recommendation for "
    "further imaging sat in an unreviewed queue for 7 weeks.",
    "Delayed recognition of compartment pressures after a crush injury because analgesia "
    "requirement was attributed to anxiety.",
    "Upper GI bleed with a shock index above 1 waited overnight for endoscopy; the out-of-"
    "hours rota was on-call from home with a 60-minute call-in.",
    "Pt with new confusion in a care home was managed as delirium for 4 days before glucose "
    "was checked. HHS on presentation to ED.",
    "Necrotising fasciitis initially treated as cellulitis. LRINEC not calculated. Time from "
    "presentation to theatre 19 hours.",
    "Antenatal reduced fetal movements — third presentation in a week, CTG each time, no "
    "growth scan arranged until the fourth attendance.",
    "PE diagnosis delayed by a negative d-dimer used outside its intended pre-test-probability "
    "band. CTPA eventually positive with clot burden.",
    "Escalation to critical care requested at 1600, bed available at 0300. Organ support "
    "started 11 hours after it was indicated.",
    "Paediatric limp seen twice and discharged; septic arthritis of the hip diagnosed on the "
    "third visit. Kocher criteria not applied at either earlier attendance.",
    "Radiology discrepancy meeting picked up a missed lung apex opacity on an ED chest film "
    "from 5 months prior. Interval growth on the staging scan.",
]

FAILURE_MODES: dict[str, list[str]] = {
    "Medication Errors": MEDICATION_ERRORS,
    "Surgical Complications": SURGICAL_COMPLICATIONS,
    "Communication Breakdowns (Handoffs)": COMMUNICATION_HANDOFF,
    "Equipment / Device Failures": EQUIPMENT_DEVICE_FAILURE,
    "Diagnostic / Intervention Delays": DIAGNOSTIC_INTERVENTION_DELAY,
}

DEPARTMENTS = [
    "General Surgery",
    "Acute Medicine",
    "Emergency Department",
    "Intensive Care",
    "Orthopaedics",
    "Obstetrics & Gynaecology",
    "Cardiology",
    "Anaesthetics",
    "Paediatrics",
    "Oncology",
]


def build_frame(rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Assemble the mock M&M minutes as a DataFrame.

    Rows are drawn round-robin across the five failure modes so the corpus is balanced,
    then shuffled so ordering carries no signal.
    """
    rng = random.Random(seed)

    pool: list[tuple[str, str]] = []
    for mode, summaries in FAILURE_MODES.items():
        for text in summaries:
            pool.append((mode, text))

    if rows <= len(pool):
        chosen = rng.sample(pool, rows)
    else:
        # Top up by re-drawing; each duplicate gets a distinguishing clinical tail so no
        # two Case_Summary values are byte-identical.
        chosen = pool[:]
        tails = [
            " Reviewed at directorate governance; action plan owner assigned.",
            " Duty of candour discussion completed with the family.",
            " Datix submitted; graded moderate harm on initial triage.",
            " Similar theme to a case discussed two months ago; trend flagged.",
            " Referred for a structured judgement review.",
        ]
        while len(chosen) < rows:
            mode, text = rng.choice(pool)
            chosen.append((mode, text + rng.choice(tails)))
        rng.shuffle(chosen)

    start = dt.date(2025, 1, 6)
    records = []
    for i, (mode, text) in enumerate(chosen, start=1):
        records.append(
            {
                "Case_ID": f"MM-2025-{i:03d}",
                "Date": start + dt.timedelta(days=rng.randint(0, 364)),
                "Department": rng.choice(DEPARTMENTS),
                "Case_Summary": text,
                "Severity_Score": rng.choices(
                    [1, 2, 3, 4, 5], weights=[10, 22, 32, 24, 12], k=1
                )[0],
                # Ground truth is retained in-memory for tests but dropped before writing.
                "_ground_truth": mode,
            }
        )

    df = pd.DataFrame.from_records(records)
    return df.sort_values("Date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate mock M&M minutes.")
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="mock_mm_minutes.xlsx")
    args = ap.parse_args()

    df = build_frame(rows=args.rows, seed=args.seed)
    truth_counts = df["_ground_truth"].value_counts().to_dict()

    export = df.drop(columns=["_ground_truth"])
    export.to_excel(args.out, index=False, sheet_name="M&M Minutes")

    print(f"[mmonfar.] wrote {len(export)} rows -> {args.out}")
    print("[mmonfar.] ground-truth balance (not written to the sheet):")
    for mode, n in sorted(truth_counts.items()):
        print(f"           {n:>3}  {mode}")
    print("[mmonfar.] columns:", ", ".join(export.columns))


if __name__ == "__main__":
    main()
