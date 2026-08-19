# netgem
Correlation and analysis of SUBTYPE proximity (opposite of gemnet) for SPiNS lab 2026

### corr.py
Correlates all subtypes for a given concept (gemini data). Can process any dimensionality

### corr_feat.py 
Correlates the concept vectors (columns) in feature-strengths.xlsx (2019 human data) to identify what concepts are close to eachother in semantic space. \
Normalizes to (-1, 1)

### corr_strength.py 
-- 

### corr_human.py
correlates the subtype vectors (columns) against each other for each concept sheet in the round2networks-2026.xlsx workbook.

### sem_map_feat.py 
visualize the concept x concept similarity matrix computed by corr.py using multidimensional scaling

### wordgen-exp.py
--
