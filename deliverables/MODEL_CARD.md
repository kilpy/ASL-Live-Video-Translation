# Model Card: <project name>, <model name>

Sections follow Mitchell et al. (2019), *Model Cards for Model Reporting*, FAT* '19, Section 4. https://doi.org/10.1145/3287560.3287596

Evaluation type: exploratory (Barocas et al., 2021). Every rate carries its count.

## 1. Model details
- Team: Brent Garey, Bhargav Singapuri, Ashley Wu, and Paschal Corrigan
- Date: November 10, 2026
- Version: v0 (Week 8 baseline)
- Commit: 6f77f24
- Model type: baseline: GRU Encoder, Transformer Decoder
- Training algorithm, parameters, features: SGD with AdamW optimizer, and Cross Entropy Loss. Features have shape B batch, T frames, L tokens, F features per fram (49 points x 2 coords + 3 detection flags = 101)
- License: Data License CC BY-NC 4.0
- Questions to: Any questions can be added as issues on this github repository.

## 2. Intended use
- Primary intended uses: Translating Live ASL Speech to English
- Primary intended users: ASL Speakers
- Out-of-scope uses: Vocabulary that exists outside of our reduced ~1,000 word set

## 3. Factors
- Relevant factors (foreseeable factors that affect performance, and how they were identified): The creator of the how2sign dataset acknowledge that there is an inherent bias in the data towards lighter skinned individuals. Relevant factors we will take into account are the skin tone and gender of the signer. 

- Evaluation factors (factors reported below, and why; why they differ from the relevant factors): The how2sign dataset has a relatively even distribution based on gender (6 male and 5 female) so we can evaluate results on each of these subsets separately to determine bias.

| Stakeholder | Type (group / instrumentation / environment) | Groups | How identified | n per group |
|---|---|---|---|---|
Signer | I | Poor Lighting/ Camera Quality| Visually, metadata| Not measurable yet|
Signer | G | Minorities: Skin tone/ Gender | Visually, metadata | Currently unknown|
Signer | E | Camera Quality | Blur Level | Currently unknown

## 4. Metrics
- Performance measure: BLEURT
- Decision threshold: >= 0.35 on validation
- Uncertainty: (e.g. 95% bootstrap interval, B = 2000, seed 0)

- Performance measure: Confident Error Rate
- Decision threshold: <= 10% on validation
- Uncertainty: (e.g. 95% bootstrap interval, B = 2000, seed 0)

## 5. Evaluation data
- Dataset and split: How2Sign, predefined validation/test split (88:5:7), from hugging face mirror 
- Source and version or hash: Manifest version: 1231830fc1e8d77555a245ca22353b144e168111
- Preprocessing: We are preprocessing input data with three different pose estimators (MediaPipe, RTMW, DWPose).
We are using 7 points from the upper body (nose, shoulders, elbows wrists) and 21 points from each hands. We are then using 3 boolean flags to detect if the media pipe features were detected from the (upper body, left, hand, right hand.)

## 6. Training data
- Dataset and split: How2Sign, predefined training split (88:5:7), from hugging face mirror 
- Source and version or hash: Manifest version: 1231830fc1e8d77555a245ca22353b144e168111
- Preprocessing: The preprocessing is the same as for described for the evaluation data.

## 7. Quantitative analyses

(We do not have the data for this yet on our larger data set)

### Unitary results

| Factor | Group | Count | Value | 95% interval | Model version |
|---|---|---|---|---|---|
| | | /n | | [ , ] | v0 |

### Intersectional results

| Factors | Group | Count | Value | 95% interval | Model version |
|---|---|---|---|---|---|
| | | /n | | [ , ] | v0 |

## 8. Ethical considerations
- Data: The dataset includes little diversity in signer skin tone which is an acknowledged limitation.
- Human life: The tool is meant to enable people, and early versions will have acknowledged performance limitations. Users should not rely on this tool in situations where completely accurate translation is essential.
- Mitigations: We acknowledge that initial use should be scoped to within the defined ~1,000 word dataset. Additionally, we define that this is an exploratory project, not a production ready tool.
- Risks and harms: The inherent bias in the dataset may lead the model to perform worse for signers with a darker skin tone.
- Use cases: Testing signed translations within our english subset.

## 9. Caveats and recommendations
- Factors not measurable yet: The videos in our dataset have a consistent format, which used the same camera and angle for each recording. Therefore, we can not yet measure factors of type E without collecting new data. Additionally, there were 6 male and 5 female signers recorded to build this dataset, so potential bias on skin tone is likely. The dataset should be extended to include a more diverse group of signers using different cameras in the future.
- Groups too small to read: N/A
- Cross-team review notes (Week 9): N/A