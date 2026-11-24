# AeroRF-Heal — LinkedIn Launch & Defense Networking Kit

This guide turns your verified **AeroRF-Heal** project into a 4-part high-impact LinkedIn engineering series and a clean GitHub portfolio designed to bypass low-GPA automated HR filters and attract Defense R&D Technical Leads directly.

---

## 1. GitHub Upload Commands (First Time Setup)

After installing Git for Windows (`https://git-scm.com/download/win`), run these commands inside `C:\Users\serka\.gemini\antigravity\scratch\aerorf-heal`:

```powershell
git init
git add .gitignore README.md requirements.txt app.py docs src tests scripts models/aerorf_shapenet.pt outputs/phase2_ai_benchmark.json outputs/phase1_coupled_poc.html outputs/phase2_ai_report.html outputs/linkedin_demo_animation.html
git status
git commit -m "feat: AeroRF-Heal conformal X-band AESA digital twin (ANSYS MAPDL + Physics-Informed AI)"
```

*(Note: `.gitignore` automatically keeps `.venv/`, `outputs/doe_batches/`, and heavy ANSYS `.rst` binary files out of Git. If `data/aerorf_doe_dataset.npz` is small (< 20 MB), you may optionally add `data/aerorf_doe_dataset.npz` or let recruiters reproduce it via `scripts/generate_doe_dataset.py`.)*

Then create a public repository named `aerorf-heal` on GitHub and push:
```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/aerorf-heal.git
git push -u origin main
```

---

## 2. How to Record a Viral 20-Second LinkedIn Video (`Win + Alt + R`)

### Option A (Fastest & Smoothest): `outputs/linkedin_demo_animation.html`
1. Open `outputs/linkedin_demo_animation.html` in Chrome/Edge and press `F11` (full screen).
2. Press `Win + Alt + R` to start Windows Screen Recording.
3. Click **`▶ Play Demo`** once:
   - **0–9s:** Watch Held-Out ANSYS Case #39 (`front_spar_bay2`, `df=0.14`) flex under load while 2 FBG sensors are severed (`X`) — the X-band beam drifts **`-7.70°`** off-target and sidelobes spike to **`-10.4 dB`**.
   - **10–18s:** AI Self-Healing engages (`0.8 ms`) — the beam snaps straight back onto **`+15°`** (`-0.14°` residual) and sidelobes drop to **`-29.8 dB`**!
4. Press `Win + Alt + R` to stop. Trim 1–2s off start/end if needed.

### Option B (Interactive Live Cockpit): `app.py`
1. Run `.\.venv\Scripts\python.exe -m streamlit run app.py`
2. Record 20 seconds while clicking **`Random 2 Kill`** in the sidebar and switching between Held-Out Damaged cases — showing live `< 1 ms` latency, 3D wing/damage zone alarm, and 2D `(u, v)` beam restoration.

---

## 3. 4-Part LinkedIn Post Series (Publish 4–5 Days Apart)

### Post 1 — The Multiphysics Problem & V&V Proof
**Attach:** Screen capture or clip of `outputs/phase1_coupled_poc.html`

> **Why embedding an X-band AESA radar into a composite UAV wing breaks stealth — and how I verified it with ANSYS MAPDL + Python.**
>
> Most CAE portfolio projects stop at colorful single-physics contours ("CFD = Colors For Directors"). Real aerospace failures happen *between* physics domains.
>
> At X-band (`10 GHz`, $\lambda_0 \approx 30\text{ mm}$), just `19 mm` of wing deflection across a `16×16` conformal aperture creates a `0.64 λ` phase error across T/R elements.
>
> Before building anything complex, I benchmarked headless **ANSYS MAPDL (`SHELL181`)** against Euler-Bernoulli plate theory:
> ✅ Analytical vs. ANSYS tip deflection: `11.429 mm` vs `11.369 mm` (**`0.518%` error**).
>
> Then I coupled a `AS4/3501-6 [0/+45/-45/90]_s` composite UAV wingbox under `5G` maneuver lift + torsion + `+45°C` radar thermal gradient directly into a 3D conformal phased-array RF engine:
> 📉 Uncompensated beam at `θ₀ = +15°`: **`-5.04°` boresight error** (~880 m miss at 10 km!) and sidelobes spiking from `-29.85 dB` to **`-12.70 dB`**.
> 📈 With 6-bit conjugate phase compensation: restored to **`-0.12°` boresight** and **`-28.98 dB` SLL**.
>
> Next post: automating 120 multi-topology ANSYS runs in <3 minutes so AI can reconstruct this deformation *without* knowing the flight loads.
>
> **Türkçe Özet:** Konformal X-Band (`10 GHz`) İHA kanat radarında 5G manevra + termal yük altında `19 mm` sapmanın radar huzmesini `-5.04°` kaçırıp yan lobları `-12.7 dB` seviyesine bozduğunu ve 6-bit faz telafisiyle `-28.98 dB` seviyesine geri toplandığını gerçek ANSYS MAPDL (`%0.518` V&V hata) ile doğruladım.
>
> #AerospaceEngineering #DefenseTech #ANSYS #MAPDL #PhasedArray #AESA #Multiphysics #CAE #UAV

---

### Post 2 — Headless Batch ANSYS Automation (No GUI Clicking)
**Attach:** Code snippet screenshot of `src/aerorf/ansys_runner.py` + DOE summary printout

> **120 real ANSYS MAPDL composite wingbox simulations in 170 seconds (`1.42s/case`) — zero GUI clicks.**
>
> Using a single-session parametric APDL DOE engine orchestrated from Python (`AeroRF-Heal`), I simulated 4 structural topologies (`intact`, `front_spar_bay2`, `rear_spar_bay2`, `upper_skin_bay2`) across an 8D Latin Hypercube envelope of lift, torsion, localized gusts, and thermal radar hotspots.
>
> Why code-first FEA matters in defense R&D:
> 🔹 Every model exports clean `12` sparse surface FBG strain readings (`EPTO`) and full `256` AESA element 3D displacements.
> 🔹 `0` MAPDL solver errors across all 120 runs.
> 🔹 `12` POD modes capture **`100.000%`** of aperture deformation energy.
>
> Next up: teaching a Physics-Informed neural network to survive **2 severed FBG sensors + battle damage** in `< 1 ms`.
>
> **Türkçe Özet:** Arayüzde tek tek tıklamak yerine Python üzerinden tek oturumluk parametrik APDL motoruyla 4 farklı yapısal hasar topolojisinde **120 gerçek ANSYS analizini 170 saniyede (vaka başı `1.42 sn`, `0` hata)** otomatik koşturdum.
>
> #PyANSYS #ANSYS #APDL #CompositeStructures #DigitalTwin #Automation #Mechatronics #SavunmaSanayi

---

### Post 3 — Physics-Informed AI Surviving 2 Severed Combat Sensors (`AeroRFShapeNet v2`)
**Attach:** Screenshot of `outputs/phase2_ai_report.html`

> **What happens when 2 out of 12 wing strain sensors get severed in combat — and your linear estimator crashes?**
>
> On 24 held-out unseen ANSYS composite wingbox simulations (with `3%` noise + **2 dead FBG sensors** + spar/skin battle damage), standard **Linear Ridge** drops to **`2.99 mm` RMSE** because it reads dead channels as zero strain.
>
> My decoupled **`AeroRFShapeNet v2`** combines two physics priors:
> 1️⃣ **Conditional Gaussian Imputation:** Reconstructs severed FBG channels from surviving sensors via training strain covariance.
> 2️⃣ **Parity-Space Compatibility Residual + RF Phase Loss ($1 - \cos(k_0 \Delta\mathbf{r} \cdot \hat{\mathbf{u}}_0)$):** Isolates structural damage independently of G-load sign/magnitude while optimizing directly in the X-band phase plane.
>
> **Held-Out Combat Results (`3%` noise + `2` severed sensors):**
> ⚡ Inference Latency: **`0.818 ms`** (~21,700× faster than ANSYS)
> 📐 Shape RMSE: **`0.615 mm`** (**4.86× more accurate** than Ridge)
> 🛡️ Battle-Damage Classification: **`80.8%`** (87.5% clean, 89.2% noisy)
> 📡 Blind 6-Bit RF Recovery (`θ₀ = +15°`):
>   • Boresight Error: `3.16° → 0.10°` (Oracle: `0.099°`)
>   • Stealth SLL: `-17.67 dB → -29.20 dB` (Oracle: `-29.37 dB`)
>
> **Türkçe Özet:** Muharebede 12 sensörden 2'si koptuğunda klasik lineer regresyon `2.99 mm` hataya fırlarken, koşullu sensör tamamlama ve parite-uzayı hasar tespiti kullanan `AeroRFShapeNet v2` **`0.82 ms`** içinde `0.62 mm` hata ile radar sapmasını `3.16° → 0.10°` ve yan lobları `-17.67 → -29.20 dB` seviyesine onardı.
>
> #PhysicsInformedAI #DeepLearning #StructuralHealthMonitoring #Radar #AESA #DefenseR&D #PyTorch

---

### Post 4 — The Grand Finale: Live Self-Healing Video + Open-Source GitHub
**Attach:** The 20-second screen recording of `outputs/linkedin_demo_animation.html` (or `app.py`) + GitHub repo link

> **Watch an X-band conformal radar heal itself from `-7.70°` target miss & `-10.4 dB` sidelobe collapse in `0.8 ms` — while 2 sensors are dead and the front spar is damaged.**
>
> Over the past phase, I built **`AeroRF-Heal`** end-to-end as an open-source Mechatronics R&D Digital Twin combining **ANSYS MAPDL 2026 R1**, **Phased-Array Electromagnetics**, **Sparse Sensor Imputation**, **Physics-Informed PyTorch**, and a **Live Streamlit Cockpit**.
>
> Open-source GitHub repo (all V&V unit tests, batch APDL generators, PyTorch code, and Streamlit app included):
> 🔗 `<PASTE_YOUR_GITHUB_URL_HERE>`
>
> I'm a Mechatronics Engineering graduate passionate about defense R&D where **structural FEA, RF/electro-optics, smart sensing, and real-time AI** intersect. Open to R&D roles in Structural/Multiphysics Simulation, Digital Twins, Avionics/Sensor Fusion, and Defense Systems.
>
> **Türkçe Özet:** Ön iç kirişi hasarlı (`df=0.14`) ve 2 sensörü kopmuş gerçek bir ANSYS test vakasında `-7.70°` sapan X-Band radar huzmesinin **`0.8 ms`** içinde tekrar hedefe (`-0.14°`) kilitlenişinin canlı simülasyonu. Açık kaynak GitHub reposu yukarıda!
>
> #OpenSource #DigitalTwin #Mechatronics #DefenseIndustry #ASELSAN #TAI #BAYKAR #ROKETSAN #SavunmaSanayi

---

## 4. Direct Outreach Playbook (Bypassing Automated GPA Filters)

1. Wait until **Post 3 or Post 4** is live so your profile shows concrete proof immediately when someone clicks your name.
2. Search LinkedIn for **Team Leads / Principal Engineers / R&D Managers** (NOT generic HR bots) in:
   - *Yapısal Analiz / Structural Analysis* (TUSAŞ, BAYKAR, TEI)
   - *Radar / RF / Anten Mekanik & Termal Tasarım* (ASELSAN REHİS/MGEO, Meteksan, ROKETSAN)
   - *Dijital İkiz / Hesaplamalı Mühendislik / Yapay Zeka* (STM, HAVELSAN, TÜBİTAK SAGE)
3. Send a short, respectful 3-line Turkish connection note:
   > *"Merhaba [İsim] Bey/Hanım, yeni mezun Mekatronik Mühendisi olarak ANSYS MAPDL, seyrek FBG sensör tamamlama ve X-Band konformal AESA radar faz telafisini birleştiren açık kaynak bir Ar-Ge projesi (AeroRF-Heal) geliştirdim. Sektördeki tecrübeniz ışığında mimari üzerine görüşünüzü almak ve ağınıza katılmak isterim."*
