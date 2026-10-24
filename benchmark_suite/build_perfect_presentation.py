import os
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen standard
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette (IEEE Modern Professional Theme)
    C_NAVY = RGBColor(15, 23, 42)      # #0f172a
    C_HEADER_BG = RGBColor(24, 49, 83) # #183153
    C_BLUE = RGBColor(37, 99, 235)     # #2563eb
    C_GREEN = RGBColor(22, 163, 74)    # #16a34a
    C_ORANGE = RGBColor(234, 88, 12)   # #ea580c
    C_TEXT = RGBColor(30, 41, 59)      # #1e293b
    C_MUTED = RGBColor(100, 116, 139)  # #64748b
    C_BG_CARD = RGBColor(241, 245, 249)# #f1f5f9
    C_WHITE = RGBColor(255, 255, 255)

    def add_header(slide, title_text, category_text="SHORT-TERM LOAD FORECASTING (STLF)"):
        # Top banner shape
        banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.15))
        banner.fill.solid()
        banner.fill.fore_color.rgb = C_HEADER_BG
        banner.line.color.rgb = C_HEADER_BG

        # Banner text
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.733), Inches(0.95))
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(10)
        p0.font.bold = True
        p0.font.color.rgb = RGBColor(147, 197, 253)

        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = C_WHITE

    def add_footer(slide, current_idx, total_slides=26):
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(7.1), Inches(11.733), Inches(0.35))
        tf = txBox.text_frame
        tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.text = f"Group 18 — TKM College of Engineering  |  STLF-SuperEnsemble Benchmark  |  Slide {current_idx} of {total_slides}"
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_MUTED

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    # Background accent card
    card = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9))
    card.fill.solid()
    card.fill.fore_color.rgb = C_BG_CARD
    card.line.color.rgb = RGBColor(203, 213, 225)
    card.line.width = Pt(1.5)

    # Top title bar
    tbar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(1.8))
    tbar.fill.solid()
    tbar.fill.fore_color.rgb = C_HEADER_BG
    tbar.line.color.rgb = C_HEADER_BG
    tf1 = tbar.text_frame
    tf1.word_wrap = True
    p = tf1.paragraphs[0]
    p.text = "LOAD FORECASTING"
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

    p_sub = tf1.add_paragraph()
    p_sub.text = "A Convex Hybrid Ensemble (GBDT + TCN + PatchBiLSTM) Benchmarked on the London Smart Meter Grid"
    p_sub.alignment = PP_ALIGN.CENTER
    p_sub.font.size = Pt(15)
    p_sub.font.color.rgb = RGBColor(224, 231, 255)

    # Content info
    tx = s1.shapes.add_textbox(Inches(1.2), Inches(2.9), Inches(10.9), Inches(3.5))
    tf = tx.text_frame
    tf.word_wrap = True
    
    p = tf.paragraphs[0]
    p.text = "Presented by: Group 18"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = C_BLUE

    members = [
        ("AADITHYA PK", "B23CSB02"),
        ("ASHIK ALI", "B23CSB22"),
        ("SAHIL RAMZAN BHAT", "B23CSB57"),
        ("ASHIQ MUHAMED KN", "B23CSB78")
    ]
    for name, roll in members:
        p = tf.add_paragraph()
        p.text = f"• {name:<30} Roll No: {roll}"
        p.font.size = Pt(13)
        p.font.color.rgb = C_TEXT

    p_div = tf.add_paragraph()
    p_div.text = "\nDepartment of Computer Science & Engineering, TKM College of Engineering"
    p_div.font.bold = True
    p_div.font.size = Pt(13)
    p_div.font.color.rgb = C_NAVY

    p_sup = tf.add_paragraph()
    p_sup.text = "Project Supervisor: Dr. Anantha Padmanabhan N. K."
    p_sup.font.size = Pt(13)
    p_sup.font.color.rgb = C_TEXT

    # -------------------------------------------------------------
    # SLIDE 2: Structured Executive Agenda
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Executive Presentation Agenda")
    add_footer(s2, 2)

    agenda_topics = [
        ("01. Motivation & Research Gaps", "Smart grid challenges, physical non-linearity, and 5 key literature gaps."),
        ("02. Literature Survey & Benchmark Zoo", "Critique of 8 peer-reviewed models (2023–2025): why trees step & transformers smooth."),
        ("03. Proposed Architecture & Methodology", "The 3 orthogonal specialists, leak-free protocol, and 0.28s convex optimization."),
        ("04. Comprehensive Empirical Results", "Primary leaderboard, peak-week tracking, regime stress tests, and Pareto efficiency."),
        ("05. Mathematical Significance & Insights", "Diebold-Mariano tests (p < 0.0001), error correlation proof, and conclusions.")
    ]
    y_pos = 1.45
    for title, desc in agenda_topics:
        box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y_pos), Inches(11.733), Inches(0.92))
        box.fill.solid()
        box.fill.fore_color.rgb = C_BG_CARD
        box.line.color.rgb = RGBColor(203, 213, 225)
        
        tf = box.text_frame
        tf.margin_left = Inches(0.3)
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.bold = True
        p0.font.size = Pt(14)
        p0.font.color.rgb = C_BLUE

        p1 = tf.add_paragraph()
        p1.text = desc
        p1.font.size = Pt(11)
        p1.font.color.rgb = C_TEXT
        y_pos += 1.05

    # -------------------------------------------------------------
    # SLIDE 3: Introduction & Motivation
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Introduction: Short-Term Load Forecasting in Smart Grids")
    add_footer(s3, 3)

    # 2-column card layout
    c1 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.3))
    c1.fill.solid()
    c1.fill.fore_color.rgb = C_BG_CARD
    c1.line.color.rgb = RGBColor(203, 213, 225)
    tf1 = c1.text_frame
    tf1.margin_left = tf1.margin_right = Inches(0.3)
    p = tf1.paragraphs[0]
    p.text = "Role of Short-Term Load Forecasting (STLF)"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = C_BLUE

    points_intro = [
        "STLF predicts electrical power demand 30 minutes to 24 hours ahead at the grid feeder level.",
        "Essential for real-time grid balancing, dispatch scheduling, and preventing power blackouts.",
        "Crucial for renewable energy integration (solar and wind cannot be dispatched on demand; flexible grid forecasting compensates for volatility).",
        "Enables dynamic demand-response pricing and automated battery storage charging."
    ]
    for pt in points_intro:
        p = tf1.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(10)

    c2 = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.3))
    c2.fill.solid()
    c2.fill.fore_color.rgb = C_BG_CARD
    c2.line.color.rgb = RGBColor(203, 213, 225)
    tf2 = c2.text_frame
    tf2.margin_left = tf2.margin_right = Inches(0.3)
    p = tf2.paragraphs[0]
    p.text = "Operational Motivation & Emerging Challenges"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = C_GREEN

    points_moti = [
        "Smart Meter Explosion: High-frequency 30-minute smart meter feeds generate massive, volatile time-series streams.",
        "Decentralized Energy Shocks: Rapid adoption of domestic Heat Pumps and Electric Vehicles (EVs) creates sharp, unpredictable residential spikes.",
        "Cost of Prediction Errors: In the UK market, a 1% forecast error increases grid reserve balancing costs by millions of pounds annually.",
        "Computational Requirement: Utilities need models that can optimize within minutes, not complex black-boxes that take hours to retrain."
    ]
    for pt in points_moti:
        p = tf2.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 4: Problem Statement & Technical Need
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Problem Statement: The Inductive Bias Dilemma")
    add_footer(s4, 4)

    card_p = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(11.733), Inches(2.5))
    card_p.fill.solid()
    card_p.fill.fore_color.rgb = RGBColor(254, 242, 242)
    card_p.line.color.rgb = RGBColor(248, 113, 113)
    tf = card_p.text_frame
    tf.margin_left = tf.margin_right = Inches(0.3)
    p = tf.paragraphs[0]
    p.text = "THE PROBLEM: Single Models Suffer from Inductive Bias Mismatch"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = RGBColor(185, 28, 28)

    p_p1 = tf.add_paragraph()
    p_p1.text = "Aggregated residential load is non-linear, multi-frequency, and sharply peaked. Existing models face an architectural trade-off:"
    p_p1.font.size = Pt(12)
    p_p1.font.color.rgb = C_TEXT

    p_p2 = tf.add_paragraph()
    p_p2.text = "1. Tabular Decision Trees (LightGBM): Partition space via piecewise-constant steps. They capture calendar schedules well but cannot model continuous rate-of-change (dy/dt), overshooting sharp ramps."
    p_p2.font.size = Pt(11.5)
    p_p2.font.color.rgb = C_TEXT

    p_p3 = tf.add_paragraph()
    p_p3.text = "2. Deep Transformers (EMSformer, PatchTST): Softmax attention normalizes across 48 time steps, acting as a low-pass filter that dilutes and smooths sharp residential evening cooking peaks (>150 kWh MAE)."
    p_p3.font.size = Pt(11.5)
    p_p3.font.color.rgb = C_TEXT

    card_n = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.2), Inches(11.733), Inches(2.5))
    card_n.fill.solid()
    card_n.fill.fore_color.rgb = RGBColor(240, 253, 244)
    card_n.line.color.rgb = RGBColor(74, 222, 128)
    tf_n = card_n.text_frame
    tf_n.margin_left = tf_n.margin_right = Inches(0.3)
    p = tf_n.paragraphs[0]
    p.text = "THE NEED: An Orthogonal Convex Hybrid Framework"
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = RGBColor(21, 128, 61)

    points_need = [
        "A rigorous, leak-free benchmarking framework comparing recent literature baselines against strong tabular models on real smart meter feeds.",
        "An architectural fusion combining decision trees (for rule-based schedule cuts) with causal continuous wave convolutions and local patch attention.",
        "A formulation solved via bounded convex optimization in seconds, eliminating deep meta-learner instability and test-time latency."
    ]
    for pt in points_need:
        p = tf_n.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(11.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 5: UN Sustainable Development Goals (SDGs)
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Alignment with UN Sustainable Development Goals")
    add_footer(s5, 5)

    sdgs = [
        ("SDG 7: Affordable & Clean Energy", "Enables precise grid balance, minimizing expensive diesel/gas peaking plant startup and maximizing renewable energy utilization without grid curtailment.", C_BLUE),
        ("SDG 9: Industry, Innovation & Infrastructure", "Provides intelligent, low-latency AI digital infrastructure for electrical transmission systems, replacing slow legacy dispatch methods.", C_GREEN),
        ("SDG 11: Sustainable Cities & Communities", "Empowers smart municipal energy distribution, community microgrid demand-response, and local electric vehicle charging management.", C_ORANGE),
        ("SDG 13: Climate Action", "Directly reduces grid carbon emissions by preventing over-generation and thermal losses across distribution feeders.", RGBColor(220, 38, 38))
    ]
    coords = [(0.8, 1.45), (6.8, 1.45), (0.8, 4.2), (6.8, 4.2)]
    for (title, desc, col), (x, y) in zip(sdgs, coords):
        b = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(5.7), Inches(2.5))
        b.fill.solid()
        b.fill.fore_color.rgb = C_BG_CARD
        b.line.color.rgb = col
        b.line.width = Pt(1.5)
        tf = b.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(13.5)
        p.font.color.rgb = col
        p1 = tf.add_paragraph()
        p1.text = desc
        p1.font.size = Pt(11)
        p1.font.color.rgb = C_TEXT

    # -------------------------------------------------------------
    # SLIDE 6: Literature Survey Zoo (Part 1: Linear, GBDT & Mixers)
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Literature Benchmark Zoo (Part 1: Linear, Tree & Mixers)")
    add_footer(s6, 6)

    lit_p1 = [
        ("LightGBM (Ke et al., NeurIPS 2017)", 
         "Gradient Boosted Decision Trees on engineered lag/weather tables.",
         "Fast (1.6s); optimal for discrete calendar rules and tariff cutoffs.",
         "Piecewise-constant steps cannot model smooth slopes (dy/dt); 171.8 kWh max error."),
        ("DLinear (Zeng et al., AAAI 2023)",
         "Decomposition linear regression on trend and seasonal streams.",
         "Extremely simple; showed linear layers beat early Transformers.",
         "Zero non-linear modeling capacity; fails during winter cold-snaps (51.65 kWh MAE)."),
        ("TimeMixer (Wang et al., ICLR 2024)",
         "Multi-scale downsampling pyramid mixing representations across frequencies.",
         "Effective at extracting multi-frequency repeating circadian cycles.",
         "Lacks recurrent hidden memory for non-periodic holiday and storm shocks.")
    ]
    y_pos = 1.45
    for name, meth, strength, flaw in lit_p1:
        b = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y_pos), Inches(11.733), Inches(1.65))
        b.fill.solid()
        b.fill.fore_color.rgb = C_BG_CARD
        b.line.color.rgb = RGBColor(203, 213, 225)
        tf = b.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        p0 = tf.paragraphs[0]
        p0.text = name
        p0.font.bold = True
        p0.font.size = Pt(13)
        p0.font.color.rgb = C_NAVY

        p1 = tf.add_paragraph()
        p1.text = f"• Mechanism: {meth}"
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = C_TEXT

        p2 = tf.add_paragraph()
        p2.text = f"• Strength: {strength}  |  Critical Flaw: {flaw}"
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = RGBColor(185, 28, 28)
        y_pos += 1.85

    # -------------------------------------------------------------
    # SLIDE 7: Literature Survey Zoo (Part 2: Deep Transformers & Hybrids)
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Literature Benchmark Zoo (Part 2: Transformers & Hybrids)")
    add_footer(s7, 7)

    lit_p2 = [
        ("PatchTST (Nie et al., ICLR 2023)",
         "Patch-based tokenization with channel-independent self-attention.",
         "Channel independence broke physical thermodynamic load-weather coupling ($R^2=0.650$)."),
        ("EMSformer (IEEE TPWRS Nov 2025)",
         "Energy Management System multi-head attention over 48-step sequences.",
         "Softmax attention disperses mass, acting as low-pass filter that blurs evening peaks (159.9 kWh MAE)."),
        ("LTE Hybrid (IEEE TSG May 2025)",
         "Stacked LSTM + Temporal CNN + Ensemble MLP.",
         "Recurrent hidden state drift; massive 1.5-hour compute time (5,672s) impractical for 30m dispatch."),
        ("TCN-QRNN (Nature Sci. Rep. 2025)",
         "Dilated causal convolutions combined with quasi-recurrent pooling.",
         "Recency pooling bias causes lag on Sunday-Monday transitions (MAE: 35.38 kWh).")
    ]
    y_pos = 1.40
    for name, meth, flaw in lit_p2:
        b = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y_pos), Inches(11.733), Inches(1.28))
        b.fill.solid()
        b.fill.fore_color.rgb = C_BG_CARD
        b.line.color.rgb = RGBColor(203, 213, 225)
        tf = b.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        p0 = tf.paragraphs[0]
        p0.text = name
        p0.font.bold = True
        p0.font.size = Pt(12.5)
        p0.font.color.rgb = C_NAVY

        p1 = tf.add_paragraph()
        p1.text = f"• Mechanism: {meth}"
        p1.font.size = Pt(10)
        p1.font.color.rgb = C_TEXT

        p2 = tf.add_paragraph()
        p2.text = f"• Critical Flaw: {flaw}"
        p2.font.size = Pt(10)
        p2.font.bold = True
        p2.font.color.rgb = RGBColor(185, 28, 28)
        y_pos += 1.42

    # -------------------------------------------------------------
    # SLIDE 8: Research Gaps
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Identified Research Gaps in STLF Literature")
    add_footer(s8, 8)

    gaps = [
        ("Gap 1: Absence of Standardized Zero-Leakage Benchmarks", "Recent deep models evaluate on synthetic long-horizon datasets; few are tested side-by-side with tuned tabular baselines on real smart grids."),
        ("Gap 2: The Transformer Peak-Smoothing Blindspot", "Standard self-attention mechanisms diffuse probability mass, underestimating critical evening cooking/heating spikes when operational risk is highest."),
        ("Gap 3: Tree Step-Discretization on Continuous Ramps", "GBDTs output zero derivatives inside partitions, creating localized step errors during rapid morning and evening load ramps."),
        ("Gap 4: Lack of Operational Regime Stress Testing", "Most studies report only global average MAE/RMSE, hiding catastrophic failures during cold snaps, midnight valleys, and weekends."),
        ("Gap 5: The Accuracy-vs-Efficiency Trade-off", "Complex deep hybrids require hours of GPU backpropagation, violating the strict 30-minute real-time smart grid dispatch window.")
    ]
    y_pos = 1.40
    for title, desc in gaps:
        b = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y_pos), Inches(11.733), Inches(0.95))
        b.fill.solid()
        b.fill.fore_color.rgb = C_BG_CARD
        b.line.color.rgb = C_BLUE
        tf = b.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.bold = True
        p0.font.size = Pt(12.5)
        p0.font.color.rgb = C_BLUE
        p1 = tf.add_paragraph()
        p1.text = desc
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = C_TEXT
        y_pos += 1.08

    # -------------------------------------------------------------
    # SLIDE 9: Project Objectives
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Project Objectives & Scope")
    add_footer(s9, 9)

    b = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(11.733), Inches(5.3))
    b.fill.solid()
    b.fill.fore_color.rgb = C_BG_CARD
    b.line.color.rgb = RGBColor(203, 213, 225)
    tf = b.text_frame
    tf.margin_left = tf.margin_right = Inches(0.35)
    
    objs = [
        "1. Build a Reproducible Smart Grid Pipeline: Aggregate 5,567 raw household smart meter block files into a feeder-level series merged with hourly Dark Sky weather and bank holidays.",
        "2. Establish a Zero-Leakage Benchmark Protocol: Evaluate 14 total models (8 published baselines, 2 trial ablations, and 4 novel architectures) across strictly held-out winter splits.",
        "3. Design Novel Specialized Learners: Address literature flaws by developing Novel 1 (MSD-TCN) for continuous wave tracking and Novel 2 (D-PABNet) for localized patch attention.",
        "4. Propose STLF-SuperEnsemble: Fuse orthogonal decision trees and neural specialists via bounded convex optimization, converging in fractions of a second.",
        "5. Conduct Comprehensive Statistical & Regime Validation: Prove model superiority across 14 metrics, operational stress regimes (Peak, Base, Cold Snap), and Diebold-Mariano tests (p < 0.0001)."
    ]
    p = tf.paragraphs[0]
    p.text = "Core Research Objectives"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(12)

    for ob in objs:
        p = tf.add_paragraph()
        p.text = ob
        p.font.size = Pt(12.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(14)

    # -------------------------------------------------------------
    # SLIDE 10: Methodology & Data Pipeline
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Methodology: Data Pipeline & Zero-Leakage Protocol")
    add_footer(s10, 10)

    # 3 Steps cards
    steps = [
        ("Step 1: Ingestion & Aggregation", 
         "• 5,567 London household streams (half-hourly kWh) summed to feeder-level.\n• Gaps <= 4 steps (2h) time-interpolated; remaining dropped.\n• Output: 39,391 half-hourly grid steps (Nov 2011 - Feb 2014)."),
        ("Step 2: Feature Engineering (19 Inputs)", 
         "• Weather: Temperature, Apparent Temp, Humidity, Pressure.\n• Calendar & Cyclic: Hour, Day of Week, Month, sin/cos cyclical encoding.\n• Autoregressive Lags: t-1 (30m), t-2 (1h), t-48 (24h), t-336 (7d).\n• Rolling statistics: 24h rolling mean."),
        ("Step 3: Zero-Leakage Splits", 
         "• Train (70%): Nov 2011 - Jun 2013 (27,573 steps). Scalers fit here only!\n• Validation (15%): Jul 2013 - Oct 2013 (5,909 steps) for weights & early stop.\n• Test (15%): Oct 29, 2013 - Feb 28, 2014 (5,861 aligned winter steps).")
    ]
    x_pos = 0.8
    for title, desc in steps:
        b = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x_pos), Inches(1.45), Inches(3.7), Inches(5.3))
        b.fill.solid()
        b.fill.fore_color.rgb = C_BG_CARD
        b.line.color.rgb = RGBColor(203, 213, 225)
        tf = b.text_frame
        tf.margin_left = tf.margin_right = Inches(0.2)
        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(13)
        p.font.color.rgb = C_BLUE
        p.space_after = Pt(10)

        p1 = tf.add_paragraph()
        p1.text = desc
        p1.font.size = Pt(11)
        p1.font.color.rgb = C_TEXT
        p1.line_spacing = 1.3
        x_pos += 4.0

    # -------------------------------------------------------------
    # SLIDE 11: End-to-End System Architecture (EMBEDDING CRISP IMAGE)
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "End-to-End Architecture: STLF-SuperEnsemble")
    add_footer(s11, 11)

    arch_img = os.path.join(r"D:\dataset\smart london\results", "Figure_0_system_architecture.png")
    if os.path.exists(arch_img):
        s11.shapes.add_picture(arch_img, Inches(0.8), Inches(1.35), Inches(11.733), Inches(5.5))

    # -------------------------------------------------------------
    # SLIDE 12: Why the Initial Prototype Failed (Ablation Analysis)
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Trial Prototype Failure Analysis (Ablation Stage)")
    add_footer(s12, 12)

    c1 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.3))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(254, 242, 242)
    c1.line.color.rgb = RGBColor(248, 113, 113)
    tf1 = c1.text_frame
    tf1.margin_left = tf1.margin_right = Inches(0.25)
    p = tf1.paragraphs[0]
    p.text = "Trial Prototype: MS-STLFNet (43.88 kWh)"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(185, 28, 28)

    pts_fail = [
        "Architecture: Sequentially chained TimeMixer downsampling --> TCN convolution --> LightGBM Residual Booster.",
        "Ablation Test: Removing the tree booster caused error to worsen to 44.91 kWh (No-Tree Backbone).",
        "The Sequential Bottleneck: End-to-end MSE backpropagation forced early layers to optimize for global sequence average.",
        "Gradient Smoothing: TimeMixer's linear downsampling smoothed down sharp 18:00 cooking peaks before the downstream TCN ever saw the representation.",
        "The Residual Failure: Handed a pre-smoothed latent representation, the tree booster could not recover lost high-frequency phase information."
    ]
    for pt in pts_fail:
        p = tf1.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(8)

    c2 = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.3))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(240, 253, 244)
    c2.line.color.rgb = RGBColor(74, 222, 128)
    tf2 = c2.text_frame
    tf2.margin_left = tf2.margin_right = Inches(0.25)
    p = tf2.paragraphs[0]
    p.text = "The Paradigm Shift: Decoupled Parallel Experts"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(21, 128, 61)

    pts_shift = [
        "Key Finding: Deep neural networks and tree models must NEVER be chained sequentially in series.",
        "Parallel Specialization: Each learner must inspect raw physical inputs directly without intermediate latent degradation.",
        "Inductive Bias Separation: Allow the tree to focus purely on calendar logic while convolutional filters focus purely on continuous wave dynamics.",
        "Convex Decoupling: Solve combination weights post-hoc using convex quadratic programming, eliminating neural meta-learner instability.",
        "Outcome: This architectural pivot directly yielded our Champion (21.64 kWh MAE)."
    ]
    for pt in pts_shift:
        p = tf2.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 13: The Novel Model Family (From Exploration to Champion)
    # -------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "The 4 Novel Architectures Developed in This Work")
    add_footer(s13, 13)

    novels = [
        ("Novel 1: MSD-TCN (28.55 kWh)", "TimeMixer + TCN + DLinear", "Multi-scale downsampling + dilated causal convolutions (d=1,2,4,8). First deep model to beat all literature baselines (<30 kWh). Selected as Champion Wave Specialist.", C_GREEN),
        ("Novel 2: D-PABNet (30.88 kWh)", "PatchTST + BiLSTM + Decomp", "Trend-seasonal decomposition + 2h localized patch attention (preventing peak dilution) + BiLSTM smoothing. Selected as Champion Patch Specialist.", C_BLUE),
        ("Novel 3: H-MoE-Boost (30.90 kWh)", "TCN + BiLSTM + MoE Router", "3 specialized neural heads (Base, Peak, Weather) with dynamic gating. Performed well but discarded due to high error correlation with Novel 1 & 2 (Ablation baseline).", C_ORANGE),
        ("Novel 4: STLF-SuperEnsemble (21.64 kWh)", "LightGBM + Novel 1 + Novel 2", "Orthogonal convex fusion solving bounded weights in 0.28s. Achieves global state-of-the-art across all metrics and operational stress regimes.", RGBColor(220, 38, 38))
    ]
    y_pos = 1.40
    for title, combo, desc, col in novels:
        b = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y_pos), Inches(11.733), Inches(1.25))
        b.fill.solid()
        b.fill.fore_color.rgb = C_BG_CARD
        b.line.color.rgb = col
        b.line.width = Pt(1.5)
        tf = b.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        p0 = tf.paragraphs[0]
        p0.text = f"{title}   |   Ingredients: {combo}"
        p0.font.bold = True
        p0.font.size = Pt(13)
        p0.font.color.rgb = col

        p1 = tf.add_paragraph()
        p1.text = desc
        p1.font.size = Pt(10.5)
        p1.font.color.rgb = C_TEXT
        y_pos += 1.42

    # -------------------------------------------------------------
    # SLIDE 14: Algorithm & Convex Weight Formulation
    # -------------------------------------------------------------
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Ensemble Algorithm & Convex Optimization")
    add_footer(s14, 14)

    c1 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.3))
    c1.fill.solid()
    c1.fill.fore_color.rgb = C_BG_CARD
    c1.line.color.rgb = RGBColor(203, 213, 225)
    tf1 = c1.text_frame
    tf1.margin_left = tf1.margin_right = Inches(0.25)
    p = tf1.paragraphs[0]
    p.text = "Step-by-Step Training Algorithm"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = C_NAVY

    algo_steps = [
        "1. Ingestion: Stream 5,567 raw smart meters, sum load, interpolate gaps <= 4 steps.",
        "2. Features: Build 19 tabular features; drop 7-day warmup.",
        "3. Scaling: Fit StandardScaler on Train split (27,573 steps) exclusively.",
        "4. Base Training: Train LightGBM (1.6s), Novel 1 MSD-TCN (434s), and Novel 2 D-PABNet (188s) with early stopping on validation.",
        "5. Residual Collection: Gather out-of-fold validation predictions y_val from the 3 models.",
        "6. Convex Optimization: Solve bounded weights on validation fold via L-BFGS-B (0.28s).",
        "7. Final Evaluation: Apply fixed weights to held-out test predictions (5,861 steps)."
    ]
    for st in algo_steps:
        p = tf1.add_paragraph()
        p.text = st
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(6)

    c2 = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.3))
    c2.fill.solid()
    c2.fill.fore_color.rgb = C_BG_CARD
    c2.line.color.rgb = RGBColor(203, 213, 225)
    tf2 = c2.text_frame
    tf2.margin_left = tf2.margin_right = Inches(0.25)
    p = tf2.paragraphs[0]
    p.text = "Mathematical Formulation & Weights"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = C_BLUE

    p = tf2.add_paragraph()
    p.text = "Optimization Objective (L-BFGS-B Solver):"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = C_NAVY

    p = tf2.add_paragraph()
    p.text = "min_w  (1/M) * ∑ | y_val - (w1*y_lgb + w2*y_tcn + w3*y_pab) |\nsubject to:   w1 + w2 + w3 = 1.0,   w_i >= 0"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(147, 51, 234)

    p = tf2.add_paragraph()
    p.text = "\nLearned Optimal Weight Distribution:"
    p.font.bold = True
    p.font.size = Pt(11)
    p.font.color.rgb = C_NAVY

    p = tf2.add_paragraph()
    p.text = "• LightGBM (Tabular Specialist)   : 49.83%\n• Novel 1 (MSD-TCN Wave Specialist) : 32.60%\n• Novel 2 (D-PABNet Patch Specialist): 17.57%"
    p.font.size = Pt(11.5)
    p.font.color.rgb = C_TEXT
    p.line_spacing = 1.3

    p = tf2.add_paragraph()
    p.text = "\nWhy Convex Optimization?\nBecause the objective is convex and low-dimensional (3 weights), convergence is guaranteed in 0.28 seconds with ZERO risk of local minima or meta-overfitting."
    p.font.size = Pt(10)
    p.font.color.rgb = C_MUTED

    # -------------------------------------------------------------
    # SLIDE 15: Primary Benchmark Leaderboard (Clean Styled Table)
    # -------------------------------------------------------------
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Performance: Primary Benchmark Leaderboard (5,861 Test Steps)")
    add_footer(s15, 15)

    # Table layout
    rows, cols = 15, 8
    t_shape = s15.shapes.add_table(rows, cols, Inches(0.8), Inches(1.35), Inches(11.733), Inches(5.6))
    table = t_shape.table
    col_widths = [Inches(3.4), Inches(1.0), Inches(1.0), Inches(1.1), Inches(1.1), Inches(1.1), Inches(1.2), Inches(1.8)]
    for i, w in enumerate(col_widths):
        table.columns[i].width = w

    headers = ["Model Name", "MAE (kWh)", "RMSE", "WAPE%", "MAPE%", "R²", "Time (s)", "vs. LightGBM"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_HEADER_BG
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(10)
            p.font.bold = True
            p.font.color.rgb = C_WHITE
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    table_data = [
        ("STLF-SuperEnsemble (Ours)", "21.64", "28.46", "1.80", "1.83", "0.9950", "0.28", "+13.52% (Best)"),
        ("LightGBM (Baseline)", "25.02", "32.46", "2.08", "2.15", "0.9934", "1.60", "0.00% (Base)"),
        ("Novel 1: MSD-TCN (Ours)", "28.55", "37.00", "2.37", "2.47", "0.9915", "434.4", "-14.12%"),
        ("Novel 2: D-PABNet (Ours)", "30.88", "39.59", "2.56", "2.63", "0.9902", "188.6", "-23.41%"),
        ("Novel 3: H-MoE-Boost (Ours)", "30.90", "41.46", "2.56", "2.57", "0.9893", "198.5", "-23.49%"),
        ("TCN-QRNN (Nature 2025)", "35.38", "47.71", "2.94", "2.90", "0.9858", "447.3", "-41.41%"),
        ("TimeMixer (ICLR 2024)", "42.38", "55.66", "3.52", "3.82", "0.9807", "28.5", "-69.39%"),
        ("DLinear (AAAI 2023)", "43.42", "54.20", "3.60", "3.95", "0.9817", "13.6", "-73.55%"),
        ("MS-STLFNet (Prototype)", "43.88", "57.86", "3.64", "3.62", "0.9792", "441.5", "-75.40%"),
        ("MS-STLFNet (No Tree Ablation)", "44.91", "59.20", "3.73", "3.66", "0.9782", "441.5", "-79.50%"),
        ("LTE Hybrid (IEEE TSG 2025)", "58.02", "75.62", "4.82", "4.57", "0.9644", "5672.2", "-131.89%"),
        ("iTransformer (ICLR 2024)", "71.91", "86.46", "5.97", "6.23", "0.9535", "217.4", "-187.42%"),
        ("EMSformer (IEEE TPWRS 2025)", "84.49", "103.64", "7.01", "6.82", "0.9331", "836.0", "-237.70%"),
        ("PatchTST (ICLR 2023)", "200.72", "237.10", "16.66", "16.63", "0.6500", "719.8", "-702.26%")
    ]
    for row_idx, row_vals in enumerate(table_data, start=1):
        for col_idx, val in enumerate(row_vals):
            cell = table.cell(row_idx, col_idx)
            cell.text = val
            # Highlight Champion row in soft green
            if row_idx == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(220, 252, 231)
            # Highlight LightGBM baseline in soft blue
            elif row_idx == 2:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(239, 246, 255)
            
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(9)
                p.font.color.rgb = C_NAVY if row_idx <= 2 else C_TEXT
                if row_idx == 1:
                    p.font.bold = True
                p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

    # -------------------------------------------------------------
    # SLIDE 16: Load Tracking in Winter Peak Week (Figure 1)
    # -------------------------------------------------------------
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Performance: 7-Day Load Tracking During Winter Peak Week")
    add_footer(s16, 16)

    fig1_img = os.path.join(r"D:\dataset\smart london\results", "Figure_1_load_tracking_peak_week.png")
    if os.path.exists(fig1_img):
        s16.shapes.add_picture(fig1_img, Inches(0.8), Inches(1.35), Inches(8.0), Inches(5.5))

    # Right insights card
    c = s16.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.0), Inches(1.35), Inches(3.533), Inches(5.5))
    c.fill.solid()
    c.fill.fore_color.rgb = C_BG_CARD
    c.line.color.rgb = C_BLUE
    tf = c.text_frame
    tf.margin_left = tf.margin_right = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "Key Observations"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_BLUE

    obs_f1 = [
        "Period: Dec 23–29, 2013 (London's coldest holiday peak week).",
        "Peak Tracking: Champion (red curve) tracks actual load (black) through every 18:00 dinner spike without phase lag.",
        "EMSformer Collapse: Overshoots peak demand by 150–250 kWh due to attention dispersion.",
        "LightGBM Overshoot: Shows discrete step lag during sharp ramps, overshooting evening peaks by ~80 kWh.",
        "Holiday Adaptability: Successfully tracks atypical Christmas Day midday deviations."
    ]
    for pt in obs_f1:
        p = tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 17: Operational Regime Stress Test (EMBEDDING FIGURE 2)
    # -------------------------------------------------------------
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Performance: Operational Regime Stress Comparison")
    add_footer(s17, 17)

    fig2_img = os.path.join(r"D:\dataset\smart london\results", "Figure_2_regime_stress_comparison.png")
    if os.path.exists(fig2_img):
        s17.shapes.add_picture(fig2_img, Inches(0.8), Inches(1.35), Inches(8.0), Inches(5.5))

    c = s17.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.0), Inches(1.35), Inches(3.533), Inches(5.5))
    c.fill.solid()
    c.fill.fore_color.rgb = C_BG_CARD
    c.line.color.rgb = C_GREEN
    tf = c.text_frame
    tf.margin_left = tf.margin_right = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "Regime Stress Insights"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_GREEN

    obs_f2 = [
        "Peak Hours (17–21h): Champion achieves 23.39 kWh MAE, outperforming LightGBM (27.93) and EMSformer (159.94).",
        "Overnight Base (02–06h): Lowest valley error at 11.55 kWh (LightGBM: 14.53).",
        "Cold Snaps (<20th pct): Robust under freezing temps at 19.50 kWh (DLinear fails at 51.65).",
        "Weekend Transitions: 21.69 kWh MAE (LightGBM: 24.01).",
        "Max Error Bound: Lowest single-step outlier error (154.88 kWh vs EMSformer 383.33)."
    ]
    for pt in obs_f2:
        p = tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 18: Pareto Frontier (Accuracy vs Compute Latency - Figure 3)
    # -------------------------------------------------------------
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Efficiency: Accuracy vs. Training Latency Pareto Frontier")
    add_footer(s18, 18)

    fig3_img = os.path.join(r"D:\dataset\smart london\results", "Figure_3_pareto_accuracy_vs_efficiency.png")
    if os.path.exists(fig3_img):
        s18.shapes.add_picture(fig3_img, Inches(0.8), Inches(1.35), Inches(7.8), Inches(5.5))

    c = s18.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.8), Inches(1.35), Inches(3.733), Inches(5.5))
    c.fill.solid()
    c.fill.fore_color.rgb = C_BG_CARD
    c.line.color.rgb = C_ORANGE
    tf = c.text_frame
    tf.margin_left = tf.margin_right = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "Computational Feasibility"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_ORANGE

    obs_f3 = [
        "The Pareto Frontier: Our Champion sits at the optimal boundary coordinate: (0.28s, 21.64 kWh).",
        "Deep Learning Overhead: LTE Hybrid required 5,672s (~1.5 hours) of GPU training, yet yielded high error (58.02 kWh).",
        "Online Meta-Solving: Bounded convex optimization converges in 0.28 seconds over validation residuals.",
        "Inference Speed: Generates 30-minute forecasts in <15ms, perfectly suited for real-time utility dispatch cycles.",
        "Conclusion: Proves high accuracy does NOT require hours of neural backpropagation."
    ]
    for pt in obs_f3:
        p = tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 19: Error Residual Distributions (Boxplot - Figure 4)
    # -------------------------------------------------------------
    s19 = prs.slides.add_slide(blank_layout)
    add_header(s19, "Consistency: Error Distribution & Residual Variance")
    add_footer(s19, 19)

    fig4_img = os.path.join(r"D:\dataset\smart london\results", "Figure_4_error_residuals_distribution.png")
    if os.path.exists(fig4_img):
        s19.shapes.add_picture(fig4_img, Inches(0.8), Inches(1.35), Inches(8.0), Inches(5.5))

    c = s19.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.0), Inches(1.35), Inches(3.533), Inches(5.5))
    c.fill.solid()
    c.fill.fore_color.rgb = C_BG_CARD
    c.line.color.rgb = C_GREEN
    tf = c.text_frame
    tf.margin_left = tf.margin_right = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "Distribution Insights"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_GREEN

    obs_f4 = [
        "Lowest Median Error: Champion achieves median absolute error of 16.2 kWh (first green box).",
        "Tightest IQR: Narrowest interquartile range proves error consistency across all 5,861 test steps.",
        "Suppressed Outliers: Baseline transformers show heavy tail distributions and extreme error whiskers.",
        "Grid Reliability: Narrow error dispersion means grid operators can size spinning reserve margins much smaller."
    ]
    for pt in obs_f4:
        p = tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 20: Why Does the Ensemble Win? (Weights & Correlation - Figure 5)
    # -------------------------------------------------------------
    s20 = prs.slides.add_slide(blank_layout)
    add_header(s20, "Mechanism: Mixture Weights & Error Orthogonality Proof")
    add_footer(s20, 20)

    fig5_img = os.path.join(r"D:\dataset\smart london\results", "Figure_5_ensemble_weights_and_correlation.png")
    if os.path.exists(fig5_img):
        s20.shapes.add_picture(fig5_img, Inches(0.8), Inches(1.35), Inches(8.2), Inches(5.5))

    c = s20.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.2), Inches(1.35), Inches(3.333), Inches(5.5))
    c.fill.solid()
    c.fill.fore_color.rgb = C_BG_CARD
    c.line.color.rgb = C_BLUE
    tf = c.text_frame
    tf.margin_left = tf.margin_right = Inches(0.2)
    p = tf.paragraphs[0]
    p.text = "Why the Ensemble Wins"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_BLUE

    obs_f5 = [
        "Balanced Contributions: LightGBM (49.8%), MSD-TCN (32.6%), D-PABNet (17.6%).",
        "Error Orthogonality: Correlation between LightGBM and Deep CNN is low (r = 0.41).",
        "Mathematical Cancellation: When LightGBM steps late on a steep ramp, MSD-TCN's wave continuity offsets it.",
        "When neural networks drift on holidays, LightGBM's calendar cuts correct them.",
        "Error variance is mathematically suppressed."
    ]
    for pt in obs_f5:
        p = tf.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 21: Statistical Significance (Diebold-Mariano)
    # -------------------------------------------------------------
    s21 = prs.slides.add_slide(blank_layout)
    add_header(s21, "Validation: Diebold-Mariano Statistical Significance Tests")
    add_footer(s21, 21)

    # Table layout
    rows, cols = 9, 6
    t_shape = s21.shapes.add_table(rows, cols, Inches(0.8), Inches(1.45), Inches(11.733), Inches(4.3))
    t = t_shape.table
    c_w = [Inches(3.2), Inches(1.6), Inches(1.6), Inches(1.6), Inches(1.8), Inches(1.933)]
    for i, w in enumerate(c_w):
        t.columns[i].width = w

    headers_dm = ["Champion vs. Baseline", "Baseline MAE", "MAE Reduction", "Win Rate (%)", "DM Statistic", "p-value"]
    for j, h in enumerate(headers_dm):
        cell = t.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_HEADER_BG
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = C_WHITE
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    dm_data = [
        ("vs. LightGBM (Tabular Base)", "25.02 kWh", "13.5%", "59.8%", "11.53", "< 0.0001 (Sig.)"),
        ("vs. TCN-QRNN (Nature 2025)", "35.38 kWh", "38.8%", "67.0%", "15.59", "< 0.0001 (Sig.)"),
        ("vs. TimeMixer (ICLR 2024)", "42.38 kWh", "48.9%", "72.5%", "17.77", "< 0.0001 (Sig.)"),
        ("vs. DLinear (AAAI 2023)", "43.42 kWh", "50.2%", "76.2%", "19.69", "< 0.0001 (Sig.)"),
        ("vs. LTE Hybrid (IEEE TSG 2025)", "58.02 kWh", "62.7%", "78.6%", "17.12", "< 0.0001 (Sig.)"),
        ("vs. iTransformer (ICLR 2024)", "71.91 kWh", "69.9%", "84.6%", "25.04", "< 0.0001 (Sig.)"),
        ("vs. EMSformer (IEEE TPWRS 2025)", "84.49 kWh", "74.4%", "88.1%", "20.16", "< 0.0001 (Sig.)"),
        ("vs. PatchTST (ICLR 2023)", "200.72 kWh", "89.2%", "96.2%", "25.07", "< 0.0001 (Sig.)")
    ]
    for r_idx, r_vals in enumerate(dm_data, start=1):
        for c_idx, val in enumerate(r_vals):
            cell = t.cell(r_idx, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = C_BG_CARD
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(10)
                p.font.color.rgb = C_NAVY if c_idx == 0 else C_TEXT
                if c_idx >= 4:
                    p.font.bold = True
                    p.font.color.rgb = C_GREEN
                p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    # Subtext box
    b_txt = s21.shapes.add_textbox(Inches(0.8), Inches(5.95), Inches(11.733), Inches(0.95))
    tf = b_txt.text_frame
    p = tf.paragraphs[0]
    p.text = "Mathematical Conclusion: Every single Diebold-Mariano statistic is > 11.5 with p < 0.0001, rigorously rejecting the null hypothesis of equal forecast accuracy. The Champion's 13.5% gain over LightGBM and 39–89% gain over published deep models is statistically significant, not random noise."
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    # -------------------------------------------------------------
    # SLIDE 22: Limitations & Future Work
    # -------------------------------------------------------------
    s22 = prs.slides.add_slide(blank_layout)
    add_header(s22, "Limitations & Future Work")
    add_footer(s22, 22)

    c1 = s22.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(5.6), Inches(5.3))
    c1.fill.solid()
    c1.fill.fore_color.rgb = C_BG_CARD
    c1.line.color.rgb = RGBColor(203, 213, 225)
    tf1 = c1.text_frame
    tf1.margin_left = tf1.margin_right = Inches(0.25)
    p = tf1.paragraphs[0]
    p.text = "Identified Limitations"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(185, 28, 28)

    lims = [
        "Single Geographic Feeder: Benchmarked on Greater London; requires cross-dataset validation on North American (AMPds) and Asian grids.",
        "Single-Step Ahead Focus: Currently optimizes H = 1 (30 minutes ahead); multi-step auto-regressive rollouts (e.g., 24h day-ahead) require recursive testing.",
        "Point Forecasts Only: Model outputs expected point values rather than probabilistic confidence intervals (quantiles).",
        "Atypical Holiday Behavior: Largest errors occur on unique days like New Year's Eve midnight celebration surge (+400 kWh above historical midnights)."
    ]
    for pt in lims:
        p = tf1.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(10)

    c2 = s22.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.45), Inches(5.7), Inches(5.3))
    c2.fill.solid()
    c2.fill.fore_color.rgb = C_BG_CARD
    c2.line.color.rgb = RGBColor(203, 213, 225)
    tf2 = c2.text_frame
    tf2.margin_left = tf2.margin_right = Inches(0.25)
    p = tf2.paragraphs[0]
    p.text = "Future Research Directions"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = C_GREEN

    futs = [
        "Multi-Horizon Extension: Adapt the convex framework for direct multi-step forecasting across 6h, 12h, and 24h horizons.",
        "Probabilistic Quantile Fusion: Implement pinball loss optimization to output 10th to 90th percentile uncertainty bounds for grid reserves.",
        "Cross-Domain Transfer: Benchmark on industrial microgrids with high rooftop solar PV penetration.",
        "Streaming Deployment: Package the 0.28s solver into an edge-deployable Docker microservice for real-time SCADA integration."
    ]
    for pt in futs:
        p = tf2.add_paragraph()
        p.text = f"• {pt}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 23: Conclusion & Key Contributions
    # -------------------------------------------------------------
    s23 = prs.slides.add_slide(blank_layout)
    add_header(s23, "Conclusion & Key Contributions")
    add_footer(s23, 23)

    b = s23.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(11.733), Inches(5.3))
    b.fill.solid()
    b.fill.fore_color.rgb = C_BG_CARD
    b.line.color.rgb = C_BLUE
    b.line.width = Pt(2)
    tf = b.text_frame
    tf.margin_left = tf.margin_right = Inches(0.35)
    p = tf.paragraphs[0]
    p.text = "Summary of Thesis Findings"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(12)

    concs = [
        "1. Standardized Leak-Free Benchmark: We established an auditable 14-model benchmark on 5,567 London smart meters, exposing why pure Transformers fail on residential smart grids.",
        "2. State-of-the-Art Accuracy: The proposed STLF-SuperEnsemble establishes a new record of 21.64 kWh MAE, 28.46 kWh RMSE, and R² of 0.9950 across 5,861 winter test steps.",
        "3. Decisive Improvement: Outperforms strong tabular LightGBM by 13.5% and 7 published deep models by 39% to 89% with high statistical significance (p < 0.0001).",
        "4. Operational Robustness: Delivers superior stability across all critical operational states: Peak Hours (23.39 kWh), Overnight Base (11.55 kWh), and Cold Snaps (19.50 kWh).",
        "5. Extreme Computational Efficiency: Finds optimal convex weights in 0.28 seconds, proving high accuracy does not require hours of deep neural training."
    ]
    for c in concs:
        p = tf.add_paragraph()
        p.text = c
        p.font.size = Pt(12)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 24: References
    # -------------------------------------------------------------
    s24 = prs.slides.add_slide(blank_layout)
    add_header(s24, "Key References (Literature Benchmark Zoo)")
    add_footer(s24, 24)

    b = s24.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(11.733), Inches(5.3))
    b.fill.solid()
    b.fill.fore_color.rgb = C_BG_CARD
    b.line.color.rgb = RGBColor(203, 213, 225)
    tf = b.text_frame
    tf.margin_left = tf.margin_right = Inches(0.25)
    p = tf.paragraphs[0]
    p.text = "Benchmarked Literature Citations"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(6)

    refs = [
        "[1] G. Ke et al., 'LightGBM: A highly efficient gradient boosting decision tree,' in Proc. NeurIPS, 2017.",
        "[2] A. Zeng et al., 'Are transformers effective for time series forecasting? (DLinear),' in Proc. AAAI, 2023.",
        "[3] Y. Nie et al., 'A time series is worth 64 words: Long-term forecasting with transformers (PatchTST),' in Proc. ICLR, 2023.",
        "[4] Y. Liu et al., 'iTransformer: Inverted transformers are effective for time series forecasting,' in Proc. ICLR, 2024.",
        "[5] S. Wang et al., 'TimeMixer: Decomposable multiscale mixing for time series forecasting,' in Proc. ICLR, 2024.",
        "[6] K. Ullah et al., 'TCN-QRNN model for short term energy consumption forecasting,' Nature Sci. Rep., vol. 15, 2025.",
        "[7] Q. Fang et al., 'A novel multi-step short-term power forecasting model (LTE Hybrid),' IEEE Trans. Smart Grid, 2025.",
        "[8] G. Pentsos et al., 'EMSformer: Energy management system transformer for load forecasting,' IEEE Trans. Power Syst., 2025.",
        "[9] F. X. Diebold and R. S. Mariano, 'Comparing predictive accuracy,' J. Business & Economic Statistics, vol. 13, 1995."
    ]
    for r in refs:
        p = tf.add_paragraph()
        p.text = r
        p.font.size = Pt(10)
        p.font.color.rgb = C_TEXT
        p.space_after = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 25: Thank You Slide
    # -------------------------------------------------------------
    s25 = prs.slides.add_slide(blank_layout)
    # Background accent card
    card = s25.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.9))
    card.fill.solid()
    card.fill.fore_color.rgb = C_HEADER_BG
    card.line.color.rgb = C_HEADER_BG

    tx = s25.shapes.add_textbox(Inches(1.5), Inches(2.2), Inches(10.333), Inches(3.2))
    tf = tx.text_frame
    p = tf.paragraphs[0]
    p.text = "THANK YOU"
    p.alignment = PP_ALIGN.CENTER
    p.font.bold = True
    p.font.size = Pt(42)
    p.font.color.rgb = C_WHITE

    p1 = tf.add_paragraph()
    p1.text = "Questions & Discussion"
    p1.alignment = PP_ALIGN.CENTER
    p1.font.size = Pt(22)
    p1.font.color.rgb = RGBColor(147, 197, 253)
    p1.space_after = Pt(16)

    p2 = tf.add_paragraph()
    p2.text = "Group 18 — Department of Computer Science & Engineering, TKM College of Engineering\nCode & Benchmark Repository: github.com/aadithya-pk/preprocessed"
    p2.alignment = PP_ALIGN.CENTER
    p2.font.size = Pt(13)
    p2.font.color.rgb = RGBColor(226, 232, 240)

    out_file = r"D:\dataset\smart london\STLF_SuperEnsemble_Final_Presentation.pptx"
    prs.save(out_file)
    print(f"Successfully created presentation: {out_file}")

if __name__ == "__main__":
    create_deck()
