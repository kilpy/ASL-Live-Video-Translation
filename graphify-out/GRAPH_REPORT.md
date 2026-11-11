# Graph Report - ASL-Live-Video-Translation  (2026-11-09)

## Corpus Check
- Corpus is ~8,882 words - fits in a single context window. You may not need a graph.

## Summary
- 372 nodes · 650 edges · 24 communities (17 shown, 7 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 10 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Landmark Feature Extraction
- Sequence Translation Models
- Training Data Configuration
- EDA and Annotation Loading
- Approach Configuration
- Video Processing Pipeline
- Processed Data Manifest
- Dataset Acquisition Integrity
- Vocabulary Split Tests
- Decoder Configuration
- Encoder Configuration
- Architecture Comparison
- Dataset Analysis Findings
- Project Architecture
- Evaluation Decision Metrics
- Confidence Calibration
- Translation Approaches
- Baseline Failure Analysis
- Pipeline Deployment Risks
- Reproducibility and Smoke Tests
- Dataset Licensing
- Project Metadata

## God Nodes (most connected - your core abstractions)
1. `load_config()` - 24 edges
2. `verify()` - 17 edges
3. `files` - 15 edges
4. `record()` - 15 edges
5. `data` - 14 edges
6. `main()` - 14 edges
7. `process_video()` - 13 edges
8. `main()` - 13 edges
9. `load_annotations()` - 12 edges
10. `extract()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Vocabulary coverage curve (18% at 1,000 words)` --semantically_similar_to--> `1,000-word vocabulary filter`  [INFERRED] [semantically similar]
  reports/eda/overview.png → README.md
- `test_tokenize_lowercases_and_keeps_contractions()` --calls--> `tokenize()`  [INFERRED]
  tests/test_data.py → src/asl_translate/dataset.py
- `frame()` --indirect_call--> `tokenize()`  [INFERRED]
  tests/test_dataset.py → src/asl_translate/dataset.py
- `test_approach_resolves_encoder_and_decoder_blocks()` --calls--> `load_config()`  [EXTRACTED]
  tests/test_data.py → src/asl_translate/config.py
- `test_data_root_env_override()` --calls--> `load_config()`  [EXTRACTED]
  tests/test_data.py → src/asl_translate/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three-way architecture comparison** — deliverables_architecture_seminar_baseline, deliverables_architecture_seminar_classical, deliverables_architecture_seminar_deep_foundation [EXTRACTED 1.00]
- **Evaluation and calibration framework** — deliverables_architecture_seminar_bleurt, deliverables_architecture_seminar_confident_error_rate, deliverables_architecture_seminar_expected_calibration_error, deliverables_architecture_seminar_storyboard_b_slice [EXTRACTED 1.00]
- **How2Sign train-corpus EDA views** — reports_eda_overview_clip_duration_distribution, reports_eda_overview_sentence_length_distribution, reports_eda_overview_vocabulary_coverage_curve [EXTRACTED 1.00]

## Communities (24 total, 7 thin omitted)

### Community 0 - "Landmark Feature Extraction"
Cohesion: 0.08
Nodes (19): main(), interpolate(), reduce(), extract(), make_landmarker(), read_frames(), to_array(), crop() (+11 more)

### Community 1 - "Sequence Translation Models"
Cohesion: 0.07
Nodes (11): get_device(), set_seed(), PoseGRUEncoder, Seq2Seq, collate(), ensure_data(), evaluate(), git() (+3 more)

### Community 2 - "Training Data Configuration"
Cohesion: 0.05
Nodes (38): test, train, val, data, features, fps, n_train_sentences, n_val_sentences (+30 more)

### Community 3 - "EDA and Annotation Loading"
Cohesion: 0.11
Nodes (18): load_config(), annotation_path(), load_annotations(), tokenize(), load_splits(), main(), percentiles(), plot() (+10 more)

### Community 4 - "Approach Configuration"
Cohesion: 0.07
Nodes (29): approaches, BASELINE, CLASSICAL, DEEP, decoder, encoder, decoder, encoder (+21 more)

### Community 5 - "Video Processing Pipeline"
Cohesion: 0.15
Nodes (11): fetch(), main(), build_smoke(), extract_splits(), extraction_info(), main(), open_shard(), paths_for() (+3 more)

### Community 6 - "Processed Data Manifest"
Cohesion: 0.08
Nodes (25): extraction, mediapipe, model, opencv, files, processed/landmarks/train, processed/landmarks/val, processed/smoke/train (+17 more)

### Community 7 - "Dataset Acquisition Integrity"
Cohesion: 0.19
Nodes (13): build_splits(), build_video_index(), download_annotations(), load_video_index(), main(), manifest_key(), read_manifest(), record() (+5 more)

### Community 8 - "Vocabulary Split Tests"
Cohesion: 0.17
Nodes (11): build_vocab(), covered(), make_splits(), pick_by_video(), frame(), test_covered_drops_unknown_words_empty_text_and_bad_durations(), test_manifest_detects_changed_data(), test_pick_by_video_takes_whole_videos_densest_first_and_is_seeded() (+3 more)

### Community 9 - "Decoder Configuration"
Cohesion: 0.13
Nodes (15): frozen, pretrained, decoders, byt5, rnn, transformer, attention, cell (+7 more)

### Community 10 - "Encoder Configuration"
Cohesion: 0.17
Nodes (12): features, frozen, pretrained, encoders, dinov2_shubert, pose_gru, bidirectional, dropout (+4 more)

### Community 11 - "Architecture Comparison"
Cohesion: 0.18
Nodes (10): Pose GRU Transformer baseline, Pose GRU RNN classical model, DINOv2 SHuBERT ByT5 foundation model, ASL Translation Experiment Card, ByT5, DINOv2, How2Sign, MediaPipe Holistic (+2 more)

### Community 12 - "Dataset Analysis Findings"
Cohesion: 0.18
Nodes (10): How2Sign dataset, MediaPipe pose and hand features, Official How2Sign split design, Repeated-sentence leakage risk, Shared-signer leakage risk, 1,000-word vocabulary filter, Clip duration distribution (median 5.3s), How2Sign train annotation overview (+2 more)

### Community 13 - "Project Architecture"
Cohesion: 0.33
Nodes (5): ASL Live Video Translation, BASELINE architecture, CLASSICAL architecture, DEEP architecture, Experiment Card specification

### Community 15 - "Confidence Calibration"
Cohesion: 0.67
Nodes (3): Confident-error rate, Expected calibration error, Sentence-level confidence from token log-probabilities

### Community 16 - "Translation Approaches"
Cohesion: 0.67
Nodes (3): ASL-to-English sentence translation, CLASSICAL pose-GRU RNN approach, DEEP DINOv2 SHuBERT ByT5 approach

### Community 17 - "Baseline Failure Analysis"
Cohesion: 0.67
Nodes (3): BASELINE full result, Decoder language bias, BASELINE model miscalibration

## Knowledge Gaps
- **28 isolated node(s):** `asl-translate`, `BASELINE architecture`, `CLASSICAL architecture`, `DEEP architecture`, `BASELINE smoke test` (+23 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 191 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `data` connect `Training Data Configuration` to `Approach Configuration`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **What connects `asl-translate`, `BASELINE architecture`, `CLASSICAL architecture` to the rest of the system?**
  _28 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Landmark Feature Extraction` be split into smaller, more focused modules?**
  _Cohesion score 0.08078431372549019 - nodes in this community are weakly interconnected._
- **Why does `load_config()` connect `EDA and Annotation Loading` to `Landmark Feature Extraction`, `Sequence Translation Models`, `Video Processing Pipeline`, `Dataset Acquisition Integrity`, `Vocabulary Split Tests`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **Should `Sequence Translation Models` be split into smaller, more focused modules?**
  _Cohesion score 0.06976744186046512 - nodes in this community are weakly interconnected._
- **Should `Training Data Configuration` be split into smaller, more focused modules?**
  _Cohesion score 0.05263157894736842 - nodes in this community are weakly interconnected._
- **Should `EDA and Annotation Loading` be split into smaller, more focused modules?**
  _Cohesion score 0.10873440285204991 - nodes in this community are weakly interconnected._