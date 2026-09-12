# Sonar-Aware Acoustic Physics Validation & Candidate Ranking

**Module**: `ml/validation/`  
**Platform**: USIP (SIH26057)

---

## 1. Physical Principle of Side-Scan Sonar Shadows

Elevated objects on the seafloor obstruct acoustic waves emitted by the side-scan sonar transducer, casting an **acoustic shadow** behind the target. 
- The presence of a down-range shadow is primary physical proof of an elevated, solid 3D structure rather than flat seabed sediment reflectivity variations.
- *Terminology Note*: Acoustic shadow geometry can support object-height estimation when acquisition geometry permits (i.e. when towfish altitude, slant range, and grazing angle are known).

---

## 2. Validation Metrics Computed

For every candidate bounding box $(x, y, w, h)$:
1. **Specular Highlight Mean**: Mean acoustic intensity of backscatter within the candidate box.
2. **Acoustic Shadow Contrast Ratio**:
   $$\text{Contrast Ratio} = \frac{\mu_{\text{ambient\_seabed}}}{\mu_{\text{downrange\_shadow}}}$$
   - Contrast Ratio $\ge 1.60$: **`Strong`** acoustic shadow evidence.
   - Contrast Ratio $\ge 1.25$: **`Moderate`** acoustic shadow evidence.
   - Contrast Ratio $\ge 1.10$: **`Weak`** acoustic shadow evidence.
   - Contrast Ratio $< 1.10$: **`None`** (likely flat sediment backscatter or low-relief feature).
3. **Seafloor Context Similarity**: Compares local variance inside the candidate to surrounding background sediment variance to filter ripple false alarms.

---

## 3. Multi-Factor Priority Rating

Rather than treating every detection identically, USIP computes a composite priority score:
$$\text{Priority Score} = 0.40 \cdot C_{\text{model}} + 0.35 \cdot S_{\text{shadow\_norm}} + 0.25 \cdot (1 - \text{ContextSim})$$

- **`CRITICAL`** ($\ge 0.82$): High model confidence + pronounced acoustic cast shadow + distinct from seafloor.
- **`HIGH`** ($0.65 - 0.81$): Strong physical evidence or high confidence.
- **`MEDIUM`** ($0.45 - 0.64$): Moderate signature requiring analyst verification.
- **`LOW`** ($< 0.45$): Faint return without distinct shadow.

