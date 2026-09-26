"""
fuzzy_engine.py
----------------
This is the compulsory Fuzzy Logic component - a genuine Mamdani-style fuzzy
inference system built with scikit-fuzzy (skfuzzy), not if/else rules.

Pipeline implemented here:
  1. FUZZIFICATION   - crisp inputs (sleep hours, screen time, stress) are
                        converted into degrees of membership in linguistic
                        sets (low/moderate/high) via triangular/trapezoidal
                        membership functions.
  2. RULE EVALUATION  - a bank of fuzzy IF-THEN rules (using fuzzy AND/OR)
                        fires with a strength derived from the fuzzified inputs.
  3. AGGREGATION      - rule outputs are combined into a single fuzzy output set.
  4. DEFUZZIFICATION  - centroid method converts the fuzzy output set back into
                        one crisp "digital wellness burnout risk" score (0-100).
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------- 1. Universe of discourse + Antecedents/Consequent ----------

sleep = ctrl.Antecedent(np.arange(0, 12.01, 0.1), "sleep")
screen_time = ctrl.Antecedent(np.arange(0, 16.01, 0.1), "screen_time")
stress = ctrl.Antecedent(np.arange(0, 10.01, 0.1), "stress")
risk = ctrl.Consequent(np.arange(0, 100.01, 1), "risk")

# ---------- 2. Membership functions (fuzzification shapes) ----------

sleep["low"] = fuzz.trapmf(sleep.universe, [0, 0, 3, 6])
sleep["moderate"] = fuzz.trimf(sleep.universe, [4, 6.5, 9])
sleep["high"] = fuzz.trapmf(sleep.universe, [7, 9, 12, 12])

screen_time["low"] = fuzz.trapmf(screen_time.universe, [0, 0, 2, 4])
screen_time["moderate"] = fuzz.trimf(screen_time.universe, [3, 6, 9])
screen_time["high"] = fuzz.trapmf(screen_time.universe, [7, 10, 16, 16])

stress["low"] = fuzz.trapmf(stress.universe, [0, 0, 2, 4])
stress["moderate"] = fuzz.trimf(stress.universe, [3, 5, 7])
stress["high"] = fuzz.trapmf(stress.universe, [6, 8, 10, 10])

risk["low"] = fuzz.trapmf(risk.universe, [0, 0, 20, 40])
risk["moderate"] = fuzz.trimf(risk.universe, [30, 50, 70])
risk["high"] = fuzz.trapmf(risk.universe, [60, 80, 100, 100])

# ---------- 3. Rule base (genuine fuzzy rules, fuzzy AND/OR, not crisp if-else) ----------

rules = [
    ctrl.Rule(sleep["low"] & stress["high"], risk["high"]),
    ctrl.Rule(screen_time["high"] & stress["high"], risk["high"]),
    ctrl.Rule(sleep["low"] & screen_time["high"], risk["high"]),
    ctrl.Rule(sleep["high"] & stress["low"], risk["low"]),
    ctrl.Rule(screen_time["low"] & stress["low"], risk["low"]),
    ctrl.Rule(sleep["high"] & screen_time["low"], risk["low"]),
    ctrl.Rule(sleep["moderate"] & screen_time["moderate"] & stress["moderate"], risk["moderate"]),
    ctrl.Rule(sleep["low"] & screen_time["moderate"] & stress["moderate"], risk["moderate"]),
    ctrl.Rule(sleep["moderate"] & stress["high"], risk["high"]),
    ctrl.Rule(sleep["moderate"] & stress["low"] & screen_time["moderate"], risk["low"]),
    ctrl.Rule(screen_time["high"] & sleep["moderate"], risk["moderate"]),
    ctrl.Rule(stress["moderate"] & sleep["high"], risk["low"]),
]

_risk_ctrl_system = ctrl.ControlSystem(rules)


def compute_burnout_risk(sleep_hours: float, screen_time_hours: float, stress_level: float) -> dict:
    """
    Runs the full fuzzy pipeline (fuzzify -> evaluate rules -> aggregate ->
    defuzzify via centroid) and returns the crisp risk score plus a
    human-readable category.

    Returns:
        {
          "score": float (0-100),
          "category": "Low" | "Moderate" | "High",
          "memberships": {  # degree of truth per linguistic label, for transparency/UI
              "sleep": {...}, "screen_time": {...}, "stress": {...}
          }
        }
    """
    sim = ctrl.ControlSystemSimulation(_risk_ctrl_system)
    sim.input["sleep"] = float(np.clip(sleep_hours, 0, 12))
    sim.input["screen_time"] = float(np.clip(screen_time_hours, 0, 16))
    sim.input["stress"] = float(np.clip(stress_level, 0, 10))
    sim.compute()  # <-- aggregation + centroid defuzzification happen here

    score = float(sim.output["risk"])

    if score < 40:
        category = "Low"
    elif score < 65:
        category = "Moderate"
    else:
        category = "High"

    memberships = {
        "sleep": {label: float(fuzz.interp_membership(sleep.universe, sleep[label].mf, sleep_hours))
                  for label in ["low", "moderate", "high"]},
        "screen_time": {label: float(fuzz.interp_membership(screen_time.universe, screen_time[label].mf, screen_time_hours))
                         for label in ["low", "moderate", "high"]},
        "stress": {label: float(fuzz.interp_membership(stress.universe, stress[label].mf, stress_level))
                   for label in ["low", "moderate", "high"]},
    }

    return {"score": round(score, 1), "category": category, "memberships": memberships}
