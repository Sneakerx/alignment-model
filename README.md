# Fundamental Limits in Multi-Image Alignment

Jupyter Notebook zur Visualisierung und numerischen Verifikation der theoretischen Registrierungsschranken aus:

> Aguerrebere et al., *Fundamental Limits in Multi-image Alignment*, arXiv:1602.01541

## Inhalt

Das Notebook `alignment_limits.ipynb` behandelt folgende Themen:

### Theoretische Schranken
- **CRBD** — Cramér-Rao Schranke für deterministisches Bildmodell (unbekanntes und bekanntes Referenzbild)
- **CRBSw / CRBSn** — Stochastische CRB für Flachspektrum- bzw. natürliche Bilder (1/f²-Spektrum)
- **EZZB** — Extended Ziv-Zakai Schranke (erfasst alle vier SNR-Regionen)

### Parameterstudien
- Einfluss der Bildgröße Nₚ, des SNR, der Bildstruktur (Bandbreite W) und der Bildanzahl K
- Gesamtvergleich aller Schranken und Visualisierung der vier SNR-Regionen

### Anwendungsfall: Satellitenbildregistrierung
- RMSE-Untergrenze für bekannte Referenz (CRBD_kn) im für Satellitensensoren typischen Parameterbereich
- Heatmap: erreichbare Präzision in Millipixel als Funktion von Bildgröße und SNR

### Numerisches Beispiel — simuliertes 1/f²-Bild
- Erzeugung eines natürlich wirkenden Bildes via 1/f²-Spektrum
- Exakter Sub-Pixel-Shift via **Fourier-Shift-Theorem** (keine räumliche Interpolation)
- Schätzalgorithmus: FFT-Kreuzkorrelation + **Parabolischer Peak-Fit**
- Monte-Carlo-Verifikation (300 Trials × 55 SNR-Punkte): empirischer RMSE vs. theoretische Schranken
- Zwei Regime sichtbar: *rauschbegrenzt* (nahe an CRB) und *biasbegrenzt* (Parabolnäherung)

### Reales Beispiel — Meteosat Second Generation (MSG2/SEVIRI)
- Einlesen von MSG Level-1B Native-Format-Dateien (`.nat`) mit `satpy`
- Automatische Suche nach wolkenfreiem Ausschnitt (128×128 Pixel, IR_087 bei 8.7 µm)
- Rauschschätzung aus zeitlicher Differenz zweier aufeinanderfolgender Scans (Δt = 15 min)
- Monte-Carlo-Experiment mit realer Bildtextur und realistischem Rauschpegel
- Reale Kreuzkorrelation zwischen den beiden Zeitschritten

### SNR-Konversion und Detektionsgrenze
- Beziehung zwischen Paper-SNR (Gradientenenergie), Intensitäts-SNR und Detektions-SNR am CC-Peak
- Fehlerwahrscheinlichkeit als Funktion von SNR_paper: Matching funktioniert weit unterhalb von 0 dB,
  weil die Kreuzkorrelation als Matched Filter das SNR um den Faktor √Nₚ anhebt

## Voraussetzungen

```bash
pip install -r requirements.txt
```

Für die MSG-Beispiele werden EUMETSAT Level-1B Native-Format-Dateien (`.nat`) benötigt,
die über [EUMETSAT Data Store](https://data.eumetsat.int) bezogen werden können.

## Ausführung

```bash
jupyter notebook alignment_limits.ipynb
```
